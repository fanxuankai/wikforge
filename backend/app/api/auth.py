"""认证相关 API 路由。

涵盖需求 §9：
- 本地注册/登录 / Refresh / OIDC authorize / OIDC callback
- 通过 ``get_current_user`` 注入 JWT 中间件，给后续业务模块复用
- 通过 ``require_admin`` 复用「仅管理员」依赖（任务 11.9 起被各 ``/api/admin``
  路由使用）
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.exceptions import ForbiddenException, UnauthorizedException
from app.core.redis import get_redis
from app.models.user import User
from app.services.auth_service import AuthService

# 与 ``app.scripts.init_db.create_admin_user`` 使用的环境变量保持一致：管理员
# 由初始化脚本播种，此处的依赖只识别该单一账号。后续接入完整 RBAC（角色/属性
# 模型）后，可把这里替换成真正的角色判定，无需修改路由层的 ``Depends``。
DEFAULT_ADMIN_EMAIL = "admin@wikforge.local"


def _resolve_admin_email() -> str:
    """每次调用时读取一次环境变量，方便测试通过 ``monkeypatch.setenv`` 覆盖。"""
    return os.environ.get("INITIAL_ADMIN_EMAIL", DEFAULT_ADMIN_EMAIL)


def is_admin_user(user: User) -> bool:
    """判断用户是否为系统管理员。

    与 :func:`require_admin` 同一套判定: 邮箱与 ``INITIAL_ADMIN_EMAIL``
    匹配即为 admin。后续引入完整 RBAC 时只需替换此函数体。
    """
    admin_email = _resolve_admin_email().strip().lower()
    user_email = (user.email or "").strip().lower()
    return bool(admin_email) and user_email == admin_email

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Request / Response Schemas ─────────────────────────────────────


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=64)
    display_name: str = Field(..., min_length=1, max_length=100)


class RegisterResponse(BaseModel):
    id: str
    email: str
    display_name: str

    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None = None

    model_config = {"from_attributes": True}


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=8, max_length=64)


class UpdateProfileRequest(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=100)


# ─── Dependencies ───────────────────────────────────────────────────


async def get_auth_service(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> AuthService:
    """构造 AuthService 实例。"""
    return AuthService(db=db, redis=redis)


async def get_current_user(
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> User:
    """JWT 验证中间件：从 Authorization 头提取并校验 access token。

    错误信息友好（401，由 :class:`UnauthorizedException` 统一返回）。
    """
    auth_header = request.headers.get("Authorization") or ""
    scheme, _, token = auth_header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedException("缺少认证令牌")
    return await auth_service.verify_access_token(token)


async def require_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    """「仅管理员」依赖。

    当前实现：把 :func:`_resolve_admin_email` 解析出的邮箱视为唯一管理员
    （由 ``app.scripts.init_db`` 在系统初始化时播种）。这是一个最小可用的
    占位实现，与项目现有的「单一管理员」模型对齐；后续引入完整 RBAC 时
    只需替换此函数体，所有 ``/api/admin/*`` 路由无需变更。

    Raises:
        UnauthorizedException: 未登录（由 ``get_current_user`` 抛出）。
        ForbiddenException: 已登录但不是管理员，返回 403。
    """
    admin_email = _resolve_admin_email().strip().lower()
    user_email = (current_user.email or "").strip().lower()
    if not admin_email or user_email != admin_email:
        raise ForbiddenException("需要管理员权限")
    return current_user


# ─── Endpoints ──────────────────────────────────────────────────────


@router.post("/register", response_model=RegisterResponse, status_code=201)
async def register(
    body: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> RegisterResponse:
    """注册新本地账号。"""
    user = await auth_service.register(
        email=body.email,
        password=body.password,
        display_name=body.display_name,
    )
    return RegisterResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name or "",
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """邮箱+密码登录，返回 token 对。"""
    token_pair = await auth_service.login(email=body.email, password=body.password)
    return TokenResponse(**token_pair)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    body: RefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """使用 Refresh Token 换发新 token 对。"""
    token_pair = await auth_service.refresh_token(body.refresh_token)
    return TokenResponse(**token_pair)


@router.get("/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """返回当前登录用户信息（用于 JWT 中间件烟雾测试 / 前端拉取用户信息）。"""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
    )


@router.get("/oidc/authorize")
async def oidc_authorize(
    auth_service: AuthService = Depends(get_auth_service),
) -> RedirectResponse:
    """跳转到 OIDC 提供商的授权端点。

    未配置 OIDC 时返回 422，由全局异常处理器封装。
    """
    uri = await auth_service.build_oidc_authorize_url()
    return RedirectResponse(url=uri)


@router.get("/oidc/callback", response_model=TokenResponse)
async def oidc_callback(
    code: str,
    auth_service: AuthService = Depends(get_auth_service),
) -> TokenResponse:
    """OIDC 回调：换 token、自动绑定/创建用户、签发本地 JWT。"""
    token_pair = await auth_service.handle_oidc_callback(code)
    return TokenResponse(**token_pair)


@router.post("/change-password", status_code=204)
async def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """修改当前用户的密码。

    需要正确的旧密码; 新密码强度由 Pydantic Schema 校验 (8-64 字符)。
    成功后不刷新 JWT, 客户端如需可以重新登录获得新 token。
    """
    from app.core.security import hash_password, verify_password
    from app.core.exceptions import UnauthorizedException, ValidationException

    # OIDC 账号 (本地无密码) 不能用此端点改密码
    if not current_user.password_hash:
        raise ValidationException("OIDC 账号请到身份提供方修改密码")

    if not verify_password(body.current_password, current_user.password_hash):
        raise UnauthorizedException("当前密码不正确")

    if body.current_password == body.new_password:
        raise ValidationException("新密码不能与当前密码相同")

    current_user.password_hash = hash_password(body.new_password)
    await db.flush()
    return None


@router.patch("/me", response_model=UserResponse)
async def update_me(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """更新当前用户的可编辑字段 (目前仅 display_name)。"""
    if body.display_name is not None:
        current_user.display_name = body.display_name
    await db.flush()
    await db.refresh(current_user)
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        display_name=current_user.display_name,
    )
