import os
import re
import json
import base64
import netaddr
from typing import List, AsyncGenerator, Optional
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends, Query, status
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings

from sqlalchemy import String, Integer, DateTime, ForeignKey, Text, Boolean, Index
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.future import select

from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from google.auth.transport import requests as google_requests
from jose import jwt, JWTError

# Allow HTTP for local dev; disable in production (HTTPS required)
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'


# --- 1. Configuration & Settings ---
class Settings(BaseSettings):
    DATABASE_URL: str = 'postgresql+asyncpg://postgres:postgres@localhost:5432/momo_decoder'
    GOOGLE_CLIENT_ID: str = 'your-google-client-id.apps.googleusercontent.com'
    GOOGLE_CLIENT_SECRET: str = 'your-google-client-secret'
    GOOGLE_REDIRECT_URI: str = 'http://localhost:8000/auth/google/callback'
    SECRET_KEY: str = 'super-secret-key-change-this-in-production'
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    VPN_ENFORCEMENT_ENABLED: bool = True
    ALLOWED_VPN_SUBNETS: List[str] = ['127.0.0.1/32', '10.8.0.0/24', '192.168.1.0/24']

    class Config:
        env_file = '.env'
        extra = 'forbid'


settings = Settings()

# --- 2. Database Models ---
engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    google_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str | None] = mapped_column(String(50), unique=True, index=True, nullable=True)
    theme: Mapped[str] = mapped_column(String(20), default='Default')
    notifications_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class UserHistory(Base):
    __tablename__ = 'user_history'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    original_text: Mapped[str] = mapped_column(Text, nullable=False)
    jumbled_code: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class AuditLog(Base):
    __tablename__ = 'audit_logs'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    username: Mapped[str | None] = mapped_column(String(50), nullable=True)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    endpoint: Mapped[str] = mapped_column(String(255), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    details: Mapped[str | None] = mapped_column(Text, nullable=True)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# --- 3. Pydantic Schemas ---
class UsernameSetupRequest(BaseModel):
    google_id: str
    username: str = Field(..., min_length=3, max_length=20, pattern='^[a-zA-Z0-9_]+$')


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'
    username: str


class CreateCookRequest(BaseModel):
    text: str

    @field_validator('text')
    def validate_english_only(cls, value: str):
        if not re.match(r'^[a-zA-Z\s]+$', value):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={'message': 'Use only Tasty Ingridients', 'color': 'red'}
            )
        return value


class FindDecodeRequest(BaseModel):
    jumbled_code: str


class SettingsUpdateRequest(BaseModel):
    theme: Optional[str] = None
    notifications_enabled: Optional[bool] = None


# --- 4. Lifespan & App Setup ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title='Momo - Artificial Decoder', lifespan=lifespan)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')


# --- 5. Security & Helper Functions ---
def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({'exp': expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get('sub')
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid token identity.')
        return username
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Could not validate credentials.')


def get_google_oauth_flow() -> Flow:
    client_config = {
        'web': {
            'client_id': settings.GOOGLE_CLIENT_ID,
            'client_secret': settings.GOOGLE_CLIENT_SECRET,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'redirect_uris': [settings.GOOGLE_REDIRECT_URI],
        }
    }
    return Flow.from_client_config(
        client_config=client_config,
        scopes=[
            'https://www.googleapis.com/auth/userinfo.email',
            'https://www.googleapis.com/auth/userinfo.profile',
            'openid',
        ],
        redirect_uri=settings.GOOGLE_REDIRECT_URI,
    )


def is_ip_in_vpn_subnet(client_ip: str, allowed_subnets: list[str]) -> bool:
    try:
        ip = netaddr.IPAddress(client_ip)
        for subnet in allowed_subnets:
            if ip in netaddr.IPNetwork(subnet):
                return True
    except netaddr.core.AddrFormatError:
        return False
    return False


def scramble_to_multilang(text: str) -> str:
    raw_bytes = text.encode('utf-8')
    encoded_str = base64.b64encode(raw_bytes).decode('utf-8')
    multi_lang_map = {'A': 'α', 'B': 'β', 'C': 'ç', 'D': 'Δ', 'E': 'ε', 'F': 'φ', 'G': 'γ', 'H': 'η'}
    jumbled = ''.join(multi_lang_map.get(char, char) for char in encoded_str)
    return f'MOMO_DEC_{jumbled}'


def descramble_from_multilang(jumbled_code: str) -> str:
    if not jumbled_code.startswith('MOMO_DEC_'):
        raise HTTPException(status_code=400, detail='Invalid Momo encoded string format.')

    payload = jumbled_code.replace('MOMO_DEC_', '')
    reverse_map = {'α': 'A', 'β': 'B', 'ç': 'C', 'Δ': 'D', 'ε': 'E', 'φ': 'F', 'γ': 'G', 'η': 'H'}
    base64_str = ''.join(reverse_map.get(char, char) for char in payload)

    try:
        decoded_bytes = base64.b64decode(base64_str.encode('utf-8'))
        return decoded_bytes.decode('utf-8')
    except Exception:
        raise HTTPException(status_code=400, detail='Failed to decode Momo string.')


# --- 6. Middlewares ---
@app.middleware('http')
async def vpn_security_middleware(request: Request, call_next):
    if not settings.VPN_ENFORCEMENT_ENABLED:
        return await call_next(request)

    forwarded_for = request.headers.get('X-Forwarded-For')
    client_ip = forwarded_for.split(',')[0].strip() if forwarded_for else (request.client.host if request.client else '')

    if not is_ip_in_vpn_subnet(client_ip, settings.ALLOWED_VPN_SUBNETS):
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={'error': 'Access Denied', 'message': 'Connection must originate from an authorized VPN tunnel.'},
        )

    return await call_next(request)


@app.middleware('http')
async def audit_logger_middleware(request: Request, call_next):
    response = await call_next(request)

    if request.url.path in ['/docs', '/openapi.json', '/favicon.ico']:
        return response

    client_ip = request.client.host if request.client else 'Unknown'
    username = None
    auth_header = request.headers.get('Authorization')
    if auth_header and auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            username = payload.get('sub')
        except JWTError:
            pass

    async with AsyncSessionLocal() as db:
        log_entry = AuditLog(
            username=username,
            action=f'HTTP_{request.method}',
            endpoint=request.url.path,
            ip_address=client_ip,
            status_code=response.status_code,
            details=json.dumps({'query_params': dict(request.query_params)}),
        )
        db.add(log_entry)
        await db.commit()

    return response


# --- 7. Endpoints: Auth, Dashboard & 4 Modules with Screen Themes ---

@app.get('/auth/google/login', summary='Login Screen - Indigo Slate Theme')
def login_with_google():
    flow = get_google_oauth_flow()
    authorization_url, _ = flow.authorization_url(prompt='consent')
    return RedirectResponse(url=authorization_url)


@app.get('/auth/google/callback', summary='Google OAuth2 Callback')
async def google_callback(code: str = Query(...), db: AsyncSession = Depends(get_db)):
    try:
        flow = get_google_oauth_flow()
        flow.fetch_token(code=code)
        credentials = flow.credentials

        id_info = id_token.verify_oauth2_token(
            credentials.id_token,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
        )

        google_id = id_info.get('sub')
        email = id_info.get('email')

        result = await db.execute(select(User).where(User.google_id == google_id))
        user = result.scalars().first()

        if user:
            if user.username:
                token = create_access_token({'sub': user.username, 'google_id': user.google_id})
                return {
                    'screen_theme': {
                        'primary': '#4E54C8',
                        'secondary': '#8F94FB',
                        'background': 'linear-gradient(135deg, #1E1E2E 0%, #2D2B55 100%)',
                        'text_color': '#FFFFFF'
                    },
                    'status': 'authenticated',
                    'access_token': token,
                    'username': user.username
                }
            return {
                'screen_theme': {
                    'primary': '#6C5CE7',
                    'secondary': '#A29BFE',
                    'background': 'linear-gradient(135deg, #130F40 0%, #30336B 100%)',
                    'text_color': '#FFFFFF'
                },
                'status': 'requires_username',
                'google_id': google_id,
                'email': email
            }

        new_user = User(google_id=google_id, email=email, username=None)
        db.add(new_user)
        await db.commit()

        return {
            'screen_theme': {
                'primary': '#6C5CE7',
                'secondary': '#A29BFE',
                'background': 'linear-gradient(135deg, #130F40 0%, #30336B 100%)',
                'text_color': '#FFFFFF'
            },
            'status': 'requires_username',
            'google_id': google_id,
            'email': email
        }

    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Authentication failed.')


@app.post('/auth/setup-username', response_model=TokenResponse)
async def setup_username(payload: UsernameSetupRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.google_id == payload.google_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Google account not found.')

    username_check = await db.execute(select(User).where(User.username == payload.username))
    if username_check.scalars().first():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='Username is taken.')

    user.username = payload.username
    await db.commit()

    access_token = create_access_token({'sub': user.username, 'google_id': user.google_id})
    return TokenResponse(access_token=access_token, username=user.username)


@app.get('/dashboard/stage', summary='Main Stage Hub - Emerald Cyan Theme')
async def get_dashboard_stage(current_user: str = Depends(get_current_user)):
    return {
        'screen_theme': {
            'primary': '#11998E',
            'secondary': '#38EF7D',
            'background': 'linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%)',
            'card_background': '#1B2A32',
            'text_color': '#FFFFFF'
        },
        'stage_name': 'Main Stage',
        'username': current_user,
        'boxes': [
            {'id': 'box_history', 'name': 'History', 'accent_color': '#4CA1AF'},
            {'id': 'box_create', 'name': 'Create', 'accent_color': '#D76D77'},
            {'id': 'box_find', 'name': 'Find', 'accent_color': '#004E92'},
            {'id': 'box_settings', 'name': 'Settings', 'accent_color': '#928DAB'}
        ]
    }


# --- Box 1: History (Royal Velvet Violet Theme) ---
@app.get('/box/history', summary='Box 1: History Screen - Velvet Theme')
async def get_history(
    current_user: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserHistory).where(UserHistory.username == current_user))
    records = result.scalars().all()

    screen_theme = {
        'theme_name': 'Royal Velvet Violet',
        'primary_accent': '#4CA1AF',
        'secondary_accent': '#2C3E50',
        'background': 'linear-gradient(135deg, #2C3E50 0%, #4CA1AF 100%)',
        'card_bg': 'rgba(255, 255, 255, 0.1)',
        'text_color': '#F4F6F7'
    }

    if not records:
        return {
            'screen_theme': screen_theme,
            'status': 'empty',
            'message': 'No History',
            'data': []
        }

    return {
        'screen_theme': screen_theme,
        'status': 'success',
        'data': [
            {
                'id': rec.id,
                'action': rec.action_type,
                'original_text': rec.original_text,
                'jumbled_code': rec.jumbled_code,
                'copyable_text': rec.jumbled_code,
                'timestamp': rec.timestamp.isoformat()
            }
            for rec in records
        ]
    }


@app.delete('/box/history/delete', summary='Delete History Action')
async def delete_history(
    current_user: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(UserHistory).where(UserHistory.username == current_user))
    records = result.scalars().all()

    for rec in records:
        await db.delete(rec)

    await db.commit()

    return {
        'screen_theme': {
            'background': 'linear-gradient(135deg, #2C3E50 0%, #4CA1AF 100%)',
            'text_color': '#FFFFFF'
        },
        'status': 'success',
        'message': 'No History',
        'instructions': 'All previous data deleted. New data will automatically be saved until cleared.'
    }


# --- Box 2: Create (Sunset Gold & Amber Theme) ---
@app.get('/box/create/ui', summary='Box 2: Create Screen UI Metadata')
async def get_create_ui(current_user: str = Depends(get_current_user)):
    return {
        'screen_theme': {
            'theme_name': 'Warm Sunset Gold',
            'primary_accent': '#FF7E5F',
            'secondary_accent': '#FEB47B',
            'background': 'linear-gradient(135deg, #3A1C71 0%, #D76D77 50%, #FFAF7B 100%)',
            'input_box_bg': 'rgba(255, 255, 255, 0.15)',
            'text_color': '#FFFFFF'
        },
        'header_text': 'Make your delicious Momo',
        'watermark': 'Place your Momos here',
        'validation_rule': 'Only English letters allowed.',
        'error_text': 'Use only Tasty Ingridients',
        'error_color': '#FF3838',
        'button_label': 'Cook Here'
    }


@app.post('/box/create/cook', summary='Cook Here Action')
async def cook_momo_instruction(
    payload: CreateCookRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    jumbled_code = scramble_to_multilang(payload.text)

    history_entry = UserHistory(
        username=current_user,
        action_type='COOK_ENCODE',
        original_text=payload.text,
        jumbled_code=jumbled_code
    )
    db.add(history_entry)
    await db.commit()

    return {
        'screen_theme': {
            'background': 'linear-gradient(135deg, #3A1C71 0%, #D76D77 50%, #FFAF7B 100%)',
            'card_accent': '#FF7E5F',
            'text_color': '#FFFFFF'
        },
        'status': 'success',
        'result_type': 'copyable_text',
        'copy_to_clipboard': True,
        'jumbled_code': jumbled_code,
        'copyable_text': jumbled_code,
        'instruction': 'Copy this jumbled code and place it into the Find box to decode.'
    }


# --- Box 3: Find (Cyberpunk Neon Teal Theme) ---
@app.post('/box/find/result', summary='Box 3: Find Screen - Cyberpunk Theme')
async def find_and_decode(
    payload: FindDecodeRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    decoded_text = descramble_from_multilang(payload.jumbled_code)

    history_entry = UserHistory(
        username=current_user,
        action_type='FIND_DECODE',
        original_text=decoded_text,
        jumbled_code=payload.jumbled_code
    )
    db.add(history_entry)
    await db.commit()

    return {
        'screen_theme': {
            'theme_name': 'Cyberpunk Neon Teal',
            'primary_accent': '#00D2FF',
            'secondary_accent': '#3A7BD5',
            'background': 'linear-gradient(135deg, #000428 0%, #004E92 100%)',
            'input_box_bg': 'rgba(0, 210, 255, 0.1)',
            'text_color': '#E0F7FA'
        },
        'status': 'success',
        'result_type': 'copyable_text',
        'copy_to_clipboard': True,
        'decoded_text': decoded_text,
        'copyable_text': decoded_text,
        'button_label': 'Result'
    }


# --- Box 4: Settings (Dark Charcoal & Ruby Theme) ---
@app.get('/box/settings', summary='Box 4: Settings Screen - Ruby Theme')
async def get_settings_overview(
    current_user: str = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.username == current_user))
    user = result.scalars().first()

    return {
        'screen_theme': {
            'theme_name': 'Dark Charcoal & Ruby',
            'primary_accent': '#928DAB',
            'secondary_accent': '#1F1C2C',
            'background': 'linear-gradient(135deg, #1F1C2C 0%, #928DAB 100%)',
            'logout_button_color': '#FF0000',
            'text_color': '#FFFFFF'
        },
        'username': current_user,
        'sections': {
            'general': {
                'appearance': {
                    'options': ['Default', 'Light', 'Dark'],
                    'current_theme': user.theme if user else 'Default'
                }
            },
            'permissions': {
                'notifications': {
                    'status': 'ON' if (user and user.notifications_enabled) else 'OFF',
                    'instruction': 'Toggling OFF disables program processing notifications.'
                }
            },
            'about': {
                'App Name': 'Momo',
                'App Developer': 'Mohit',
                'Co-ordinator': 'Rocky',
                'Version': '1.26.45.1'
            },
            'logout': {
                'label': 'Log out',
                'color': '#FF0000',
                'action_endpoint': '/box/settings/logout'
            }
        }
    }


@app.post('/box/settings/update', summary='Update Settings Options')
async def update_settings(
    payload: SettingsUpdateRequest,
    current_user: str = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.username == current_user))
    user = result.scalars().first()

    if not user:
        raise HTTPException(status_code=404, detail='User not found.')

    if payload.theme:
        if payload.theme not in ['Default', 'Light', 'Dark']:
            raise HTTPException(status_code=400, detail='Invalid theme selection.')
        user.theme = payload.theme

    if payload.notifications_enabled is not None:
        user.notifications_enabled = payload.notifications_enabled

    await db.commit()
    return {'status': 'success', 'message': 'Settings updated successfully.', 'current_theme': user.theme}


@app.post('/box/settings/logout', summary='Red-Colored Log Out Option')
async def logout(current_user: str = Depends(get_current_user)):
    return {
        'screen_theme': {
            'background': 'linear-gradient(135deg, #1E1E2E 0%, #2D2B55 100%)'
        },
        'status': 'success',
        'message': f'User {current_user} successfully logged out.',
        'redirect': '/auth/google/login'
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)