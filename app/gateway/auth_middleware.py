"""全局认证中间件——fail-closed 安全兜底。

对非公开路径的未认证请求返回 401。请求通过 cookie 检查后,将 JWT
负载解析为真实的 ``User`` 对象,并写入 ``request.state.user`` 和
``deerflow.runtime.user_context`` contextvar,使仓储层的属主过滤
通过哨兵模式自动生效。

细粒度权限检查仍在 authz.py 的装饰器中。
"""

from collections.abc import Callable

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse
from app.gateway.auth_disabled import (
    AUTH_SOURCE_AUTH_DISABLED,
    AUTH_SOURCE_INTERNAL,
    AUTH_SOURCE_SESSION,
    get_auth_disabled_user,
    is_auth_disabled,
)
from app.gateway.authz import _ALL_PERMISSIONS, AuthContext
from app.gateway.internal_auth import INTERNAL_AUTH_HEADER_NAME, get_internal_user, is_valid_internal_auth_token
from deerflow.runtime.user_context import reset_current_user, set_current_user

# 永远不需要认证的路径。
_PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
)

# 公开的认证路径(登录/注册/状态检查)。
# /api/v1/auth/me、/api/v1/auth/change-password 等不是公开路径。
_PUBLIC_EXACT_PATHS: frozenset[str] = frozenset(
    {
        "/api/v1/auth/login/local",
        "/api/v1/auth/register",
        "/api/v1/auth/logout",
        "/api/v1/auth/setup-status",
        "/api/v1/auth/initialize",
    }
)


def _is_public(path: str) -> bool:
    stripped = path.rstrip("/")
    if stripped in _PUBLIC_EXACT_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PATH_PREFIXES)


class AuthMiddleware(BaseHTTPMiddleware):
    """严格的认证闸门:拒绝没有有效会话的请求。

    对非公开路径做两阶段检查:

    1. Cookie 是否存在——缺失时返回 401 NOT_AUTHENTICATED
    2. 通过 ``get_optional_user_from_request`` 做 JWT 校验——token 缺失、
       格式错误、过期,或签名用户不存在/已失效时返回 401 TOKEN_INVALID

    成功后写入 ``request.state.user`` 和
    ``deerflow.runtime.user_context`` contextvar,使仓储层属主过滤在
    下游自动生效,无需每个路由都加 ``@require_auth`` 装饰器。需要按资源
    授权的路由(例如"用户 A 不能通过猜 URL 读取用户 B 的 thread")应
    额外使用 ``@require_permission(..., owner_check=True)`` 做显式强制
    ——但认证本身已在此完全处理。
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if _is_public(request.url.path):
            return await call_next(request)

        internal_user = None
        if is_valid_internal_auth_token(request.headers.get(INTERNAL_AUTH_HEADER_NAME)):
            internal_user = get_internal_user()

        auth_source = AUTH_SOURCE_SESSION
        access_token = request.cookies.get("access_token")

        # 非公开路径:要求会话 cookie
        if internal_user is not None:
            user = internal_user
            auth_source = AUTH_SOURCE_INTERNAL
        elif access_token:
            # 严格 JWT 校验:对垃圾/过期 token 直接在此返回 401,而不是
            # 静默放行。这堵住了"垃圾 cookie 绕过"缺口(AUTH_TEST_PLAN
            # 测试 7.5.8):没有这一步,/api/models 这类非隔离路由会把任何
            # 形似 cookie 的字符串当作有效认证。
            #
            # 这里调用*严格*解析器,使细粒度错误码(token_expired、
            # token_invalid、user_not_found 等)能从 AuthErrorCode 传播,
            # 而不是被压平成一个通用错误码。BaseHTTPMiddleware 不会让
            # HTTPException 向上冒泡,因此在此捕获并渲染为 JSONResponse。
            from app.gateway.deps import get_current_user_from_request

            try:
                user = await get_current_user_from_request(request)
            except HTTPException as exc:
                if not is_auth_disabled():
                    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
                user = get_auth_disabled_user()
                auth_source = AUTH_SOURCE_AUTH_DISABLED
        elif is_auth_disabled():
            user = get_auth_disabled_user()
            auth_source = AUTH_SOURCE_AUTH_DISABLED
        else:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": AuthErrorResponse(
                        code=AuthErrorCode.NOT_AUTHENTICATED,
                        message="Authentication required",
                    ).model_dump()
                },
            )

        # 同时写入 request.state.user(供 contextvar 模式使用)和
        # request.state.auth(使 @require_permission 的 "auth is None"
        # 分支直接短路,避免每个请求重复执行整套 JWT 解码 + 数据库查询
        # 流程)。
        request.state.user = user
        request.state.auth_source = auth_source
        request.state.auth = AuthContext(user=user, permissions=_ALL_PERMISSIONS)
        token = set_current_user(user)
        try:
            return await call_next(request)
        finally:
            reset_current_user(token)
