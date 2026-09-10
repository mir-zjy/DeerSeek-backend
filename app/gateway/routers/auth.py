"""认证相关端点。"""

import asyncio
import logging
import os
import time
from ipaddress import ip_address, ip_network

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field, field_validator

from app.gateway.auth import (
    UserResponse,
    create_access_token,
)
from app.gateway.auth.config import get_auth_config
from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse
from app.gateway.csrf_middleware import is_secure_request
from app.gateway.deps import get_current_user_from_request, get_local_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── 请求/响应模型 ──────────────────────────────────────────────


class LoginResponse(BaseModel):
    """登录响应模型——token 仅存于 HttpOnly cookie。"""

    expires_in: int  # 秒
    needs_setup: bool = False


# 常见弱密码黑名单。取自公开的 SecLists “10k worst passwords” 集合，
# 仅保留小写且长度 >= 8 的条目（更短的已被 min_length 校验拦截）。
# 有意保持精简：这只是**最低限度**的防御，并非完整的 HIBP / passlib
# 检查，且在进程内按请求执行。
_COMMON_PASSWORDS: frozenset[str] = frozenset(
    {
        "password",
        "password1",
        "password12",
        "password123",
        "password1234",
        "12345678",
        "123456789",
        "1234567890",
        "qwerty12",
        "qwertyui",
        "qwerty123",
        "abc12345",
        "abcd1234",
        "iloveyou",
        "letmein1",
        "welcome1",
        "welcome123",
        "admin123",
        "administrator",
        "passw0rd",
        "p@ssw0rd",
        "monkey12",
        "trustno1",
        "sunshine",
        "princess",
        "football",
        "baseball",
        "superman",
        "batman123",
        "starwars",
        "dragon123",
        "master123",
        "shadow12",
        "michael1",
        "jennifer",
        "computer",
    }
)


def _password_is_common(password: str) -> bool:
    """大小写不敏感的黑名单检查。

    先将输入转小写，使 ``Password`` / ``PASSWORD`` 之类的简单变体也被拒绝。
    不做字符替换归一化（``p@ssw0rd`` 直接作为独立条目收录）——保持规则
    轻量且可预测。
    """
    return password.lower() in _COMMON_PASSWORDS


def _validate_strong_password(value: str) -> str:
    """Register 与 ChangePassword 共用的 Pydantic 字段校验函数。

    约束以函数而非类型级 mixin 实现。两个请求模型之间没有继承关系，
    只是共享同一密码强度规则。将其提取为独立函数后，各模型可通过
    ``@field_validator(field_name)`` 直接绑定，无需复杂的继承。
    """
    if _password_is_common(value):
        raise ValueError("Password is too common; choose a stronger password.")
    return value


class RegisterRequest(BaseModel):
    """用户注册请求模型。"""

    email: EmailStr
    password: str = Field(..., min_length=8)

    _strong_password = field_validator("password")(classmethod(lambda cls, v: _validate_strong_password(v)))


class ChangePasswordRequest(BaseModel):
    """修改密码请求模型（同时处理初始化设置流程）。"""

    current_password: str
    new_password: str = Field(..., min_length=8)
    new_email: EmailStr | None = None

    _strong_password = field_validator("new_password")(classmethod(lambda cls, v: _validate_strong_password(v)))


class MessageResponse(BaseModel):
    """通用消息响应。"""

    message: str


# ── 辅助函数 ───────────────────────────────────────────────────────────────


def _set_session_cookie(response: Response, token: str, request: Request) -> None:
    """在响应中设置 access_token 的 HttpOnly cookie。"""
    config = get_auth_config()
    is_https = is_secure_request(request)
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=is_https,
        samesite="lax",
        max_age=config.token_expiry_days * 24 * 3600 if is_https else None,
    )


# ── 速率限制 ────────────────────────────────────────────────────────
# 进程内字典——不会在多个 worker 之间共享。
#
# **局限性**：多 worker 部署（如 gunicorn -w N）时，每个 worker 各自
# 维护独立的锁定表，攻击者实际上可获得 N × _MAX_LOGIN_ATTEMPTS 次
# 尝试机会才会被全部锁定。生产环境多 worker 部署应改用共享存储
# （Redis、数据库计数器等）以实现真正的按 IP 限流。

_MAX_LOGIN_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300  # 5 分钟

# ip → (fail_count, lock_until_timestamp)
_login_attempts: dict[str, tuple[int, float]] = {}


def _trusted_proxies() -> list:
    """将 ``AUTH_TRUSTED_PROXIES`` 环境变量解析为 ip_network 对象列表。

    逗号分隔的 CIDR 或单个 IP 条目。为空 / 未设置 = 不信任任何代理
    （直连模式）。无效条目会跳过并记录 warning。实时读取，使环境变量
    覆盖立即生效，测试也可直接 ``monkeypatch.setenv`` 而不必触碰
    模块级缓存。
    """
    raw = os.getenv("AUTH_TRUSTED_PROXIES", "").strip()
    if not raw:
        return []
    nets = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            nets.append(ip_network(entry, strict=False))
        except ValueError:
            logger.warning("AUTH_TRUSTED_PROXIES: ignoring invalid entry %r", entry)
    return nets


def _get_client_ip(request: Request) -> str:
    """提取用于限流的真实客户端 IP。

    信任模型：

    - TCP 对端（``request.client.host``）始终作为基准值。它是内核报告的
      连接 socket，客户端自身无法伪造。
    - **仅当** TCP 对端位于 ``AUTH_TRUSTED_PROXIES`` 白名单（环境变量设置，
      逗号分隔的 CIDR 或单个 IP）内时，才采纳 ``X-Real-IP``。设置后假定
      网关位于反向代理（nginx、Cloudflare、ALB 等）之后，该代理会用
      原始客户端地址覆写 ``X-Real-IP``。
    - 未设置 ``AUTH_TRUSTED_PROXIES`` 时，静默忽略 ``X-Real-IP``——堵住
      在开发 / 直连网关模式下任何客户端都能轮换该头部来绕过按 IP 限流
      的漏洞。

    有意不使用 ``X-Forwarded-For``：它在*第一跳*天然由客户端控制，
    且信任链难以按请求审计。
    """
    peer_host = request.client.host if request.client else None

    trusted = _trusted_proxies()
    if trusted and peer_host:
        try:
            peer_ip = ip_address(peer_host)
            if any(peer_ip in net for net in trusted):
                real_ip = request.headers.get("x-real-ip", "").strip()
                if real_ip:
                    return real_ip
        except ValueError:
            # peer_host 不是可解析的 IP（如 "unknown"）——继续走默认逻辑
            pass

    return peer_host or "unknown"


def _check_rate_limit(ip: str) -> None:
    """若该 IP 当前处于锁定状态则抛出 429。"""
    record = _login_attempts.get(ip)
    if record is None:
        return
    fail_count, lock_until = record
    if fail_count >= _MAX_LOGIN_ATTEMPTS:
        if time.time() < lock_until:
            raise HTTPException(
                status_code=429,
                detail="Too many login attempts. Try again later.",
            )
        del _login_attempts[ip]


_MAX_TRACKED_IPS = 10000


def _record_login_failure(ip: str) -> None:
    """记录指定 IP 的一次登录失败。"""
    # 字典过大时清除已过期的锁定记录
    if len(_login_attempts) >= _MAX_TRACKED_IPS:
        now = time.time()
        expired = [k for k, (c, t) in _login_attempts.items() if c >= _MAX_LOGIN_ATTEMPTS and now >= t]
        for k in expired:
            del _login_attempts[k]
        # 若仍然过大，淘汰损失最小的一半：未达阈值的 IP（lock_until=0.0）
        # 排在最前，其后按锁定到期时间从最早开始淘汰。
        if len(_login_attempts) >= _MAX_TRACKED_IPS:
            by_time = sorted(_login_attempts.items(), key=lambda kv: kv[1][1])
            for k, _ in by_time[: len(by_time) // 2]:
                del _login_attempts[k]

    record = _login_attempts.get(ip)
    if record is None:
        _login_attempts[ip] = (1, 0.0)
    else:
        new_count = record[0] + 1
        lock_until = time.time() + _LOCKOUT_SECONDS if new_count >= _MAX_LOGIN_ATTEMPTS else 0.0
        _login_attempts[ip] = (new_count, lock_until)


def _record_login_success(ip: str) -> None:
    """登录成功后清除该 IP 的失败计数。"""
    _login_attempts.pop(ip, None)


# ── 端点 ─────────────────────────────────────────────────────────────


@router.post("/login/local", response_model=LoginResponse)
async def login_local(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """本地邮箱/密码登录。"""
    client_ip = _get_client_ip(request)
    _check_rate_limit(client_ip)

    user = await get_local_provider().authenticate({"email": form_data.username, "password": form_data.password})

    if user is None:
        _record_login_failure(client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=AuthErrorResponse(code=AuthErrorCode.INVALID_CREDENTIALS, message="Incorrect email or password").model_dump(),
        )

    _record_login_success(client_ip)
    token = create_access_token(str(user.id), token_version=user.token_version)
    _set_session_cookie(response, token, request)

    return LoginResponse(
        expires_in=get_auth_config().token_expiry_days * 24 * 3600,
        needs_setup=user.needs_setup,
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(request: Request, response: Response, body: RegisterRequest):
    """注册新用户账号（角色固定为 'user'）。

    首个管理员通过 /initialize 显式创建，本端点仅创建普通用户。
    通过设置会话 cookie 实现自动登录。
    """
    try:
        user = await get_local_provider().create_user(email=body.email, password=body.password, system_role="user")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=AuthErrorResponse(code=AuthErrorCode.EMAIL_ALREADY_EXISTS, message="Email already registered").model_dump(),
        )

    token = create_access_token(str(user.id), token_version=user.token_version)
    _set_session_cookie(response, token, request)

    return UserResponse(id=str(user.id), email=user.email, system_role=user.system_role)


@router.post("/logout", response_model=MessageResponse)
async def logout(request: Request, response: Response):
    """通过清除 cookie 登出当前用户。"""
    response.delete_cookie(key="access_token", secure=is_secure_request(request), samesite="lax")
    return MessageResponse(message="Successfully logged out")


@router.post("/change-password", response_model=MessageResponse)
async def change_password(request: Request, response: Response, body: ChangePasswordRequest):
    """修改当前已认证用户的密码。

    同时处理首次启动的初始化设置流程：
    - 若提供 new_email，则更新邮箱（检查唯一性）
    - 若 user.needs_setup 为 True 且提供了 new_email，则清除 needs_setup
    - 始终递增 token_version 以使旧会话失效
    - 使用新的 token_version 重新签发会话 cookie
    """
    from app.gateway.auth.password import hash_password_async, verify_password_async
    from app.gateway.auth_disabled import AUTH_SOURCE_AUTH_DISABLED

    user = await get_current_user_from_request(request)

    if getattr(request.state, "auth_source", None) == AUTH_SOURCE_AUTH_DISABLED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=AuthErrorResponse(
                code=AuthErrorCode.INVALID_CREDENTIALS,
                message="Password changes are not available when DEER_FLOW_AUTH_DISABLED=1.",
            ).model_dump(),
        )

    if user.password_hash is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=AuthErrorResponse(code=AuthErrorCode.INVALID_CREDENTIALS, message="OAuth users cannot change password").model_dump())

    if not await verify_password_async(body.current_password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=AuthErrorResponse(code=AuthErrorCode.INVALID_CREDENTIALS, message="Current password is incorrect").model_dump())

    provider = get_local_provider()

    # 若提供了新邮箱则更新
    if body.new_email is not None:
        existing = await provider.get_user_by_email(body.new_email)
        if existing and str(existing.id) != str(user.id):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=AuthErrorResponse(code=AuthErrorCode.EMAIL_ALREADY_EXISTS, message="Email already in use").model_dump())
        user.email = body.new_email

    # 更新密码并递增版本号
    user.password_hash = await hash_password_async(body.new_password)
    user.token_version += 1

    # 若处于初始化设置流程，清除设置标记
    if user.needs_setup and body.new_email is not None:
        user.needs_setup = False

    await provider.update_user(user)

    # 使用新的 token_version 重新签发 cookie
    token = create_access_token(str(user.id), token_version=user.token_version)
    _set_session_cookie(response, token, request)

    return MessageResponse(message="Password changed successfully")


@router.get("/me", response_model=UserResponse)
async def get_me(request: Request):
    """获取当前已认证用户的信息。"""
    user = await get_current_user_from_request(request)
    return UserResponse(id=str(user.id), email=user.email, system_role=user.system_role, needs_setup=user.needs_setup)


# 按 IP 缓存：ip → (时间戳, 结果字典)。
# 在 TTL 内直接返回缓存结果而非 429，因为答案（是否存在管理员）
# 很少变化，而返回 429 会破坏多标签页 / 重启后的重连风暴。
_SETUP_STATUS_CACHE: dict[str, tuple[float, dict]] = {}
_SETUP_STATUS_CACHE_TTL_SECONDS = 60
_MAX_TRACKED_SETUP_STATUS_IPS = 10000
_SETUP_STATUS_INFLIGHT: dict[str, asyncio.Task[dict]] = {}
_SETUP_STATUS_INFLIGHT_GUARD = asyncio.Lock()


@router.get("/setup-status")
async def setup_status(request: Request):
    """检查是否已存在管理员账号。无管理员时返回 needs_setup=True。"""
    client_ip = _get_client_ip(request)
    now = time.time()

    # TTL 内直接返回缓存结果——避免多标签页重连时触发 429。
    cached = _SETUP_STATUS_CACHE.get(client_ip)
    if cached is not None:
        cached_time, cached_result = cached
        if now - cached_time < _SETUP_STATUS_CACHE_TTL_SECONDS:
            return cached_result

    async with _SETUP_STATUS_INFLIGHT_GUARD:
        # 等待 inflight 锁之后重新检查缓存。
        now = time.time()
        cached = _SETUP_STATUS_CACHE.get(client_ip)
        if cached is not None:
            cached_time, cached_result = cached
            if now - cached_time < _SETUP_STATUS_CACHE_TTL_SECONDS:
                return cached_result

        task = _SETUP_STATUS_INFLIGHT.get(client_ip)
        if task is None:
            # 字典过大时清除过期条目，以限制内存占用。
            if len(_SETUP_STATUS_CACHE) >= _MAX_TRACKED_SETUP_STATUS_IPS:
                cutoff = now - _SETUP_STATUS_CACHE_TTL_SECONDS
                stale = [k for k, (t, _) in _SETUP_STATUS_CACHE.items() if t < cutoff]
                for k in stale:
                    del _SETUP_STATUS_CACHE[k]
                if len(_SETUP_STATUS_CACHE) >= _MAX_TRACKED_SETUP_STATUS_IPS:
                    by_time = sorted(_SETUP_STATUS_CACHE.items(), key=lambda entry: entry[1][0])
                    for k, _ in by_time[: len(by_time) // 2]:
                        del _SETUP_STATUS_CACHE[k]

            async def _compute_setup_status() -> dict:
                admin_count = await get_local_provider().count_admin_users()
                return {"needs_setup": admin_count == 0}

            task = asyncio.create_task(_compute_setup_status())
            _SETUP_STATUS_INFLIGHT[client_ip] = task

    try:
        result = await task
    finally:
        async with _SETUP_STATUS_INFLIGHT_GUARD:
            if _SETUP_STATUS_INFLIGHT.get(client_ip) is task:
                del _SETUP_STATUS_INFLIGHT[client_ip]

    # 仅缓存稳定的“已初始化”结果，避免过期的设置重定向。
    if result["needs_setup"] is False:
        _SETUP_STATUS_CACHE[client_ip] = (time.time(), result)
    else:
        _SETUP_STATUS_CACHE.pop(client_ip, None)
    return result


class InitializeAdminRequest(BaseModel):
    """首次启动创建管理员账号的请求模型。"""

    email: EmailStr
    password: str = Field(..., min_length=8)

    _strong_password = field_validator("password")(classmethod(lambda cls, v: _validate_strong_password(v)))


@router.post("/initialize", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def initialize_admin(request: Request, response: Response, body: InitializeAdminRequest):
    """系统初始化时创建首个管理员账号。

    仅在尚不存在管理员时可调用。若管理员已存在则返回 409 Conflict。

    成功后管理员账号以 ``needs_setup=False`` 创建，并设置会话 cookie。
    """
    admin_count = await get_local_provider().count_admin_users()
    if admin_count > 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=AuthErrorResponse(code=AuthErrorCode.SYSTEM_ALREADY_INITIALIZED, message="System already initialized").model_dump(),
        )

    try:
        user = await get_local_provider().create_user(email=body.email, password=body.password, system_role="admin", needs_setup=False)
    except ValueError:
        # DB 唯一约束竞争：另一个并发请求抢先完成。
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=AuthErrorResponse(code=AuthErrorCode.SYSTEM_ALREADY_INITIALIZED, message="System already initialized").model_dump(),
        )

    token = create_access_token(str(user.id), token_version=user.token_version)
    _set_session_cookie(response, token, request)

    return UserResponse(id=str(user.id), email=user.email, system_role=user.system_role)


# ── OAuth 端点（预留/占位）─────────────────────────────────


@router.get("/oauth/{provider}")
async def oauth_login(provider: str):
    """发起 OAuth 登录流程。

    重定向到 OAuth provider 的授权 URL。
    当前为占位实现——需要 OAuth provider 的具体实现。
    """
    if provider not in ["github", "google"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported OAuth provider: {provider}",
        )

    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="OAuth login not yet implemented",
    )


@router.get("/callback/{provider}")
async def oauth_callback(provider: str, code: str, state: str):
    """OAuth 回调端点。

    处理用户授权后 OAuth provider 的回调。
    当前为占位实现。
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="OAuth callback not yet implemented",
    )
