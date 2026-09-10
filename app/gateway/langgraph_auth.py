"""LangGraph 兼容认证处理器——与 Gateway 共享 JWT 逻辑。

DeerFlow 的默认运行时已内嵌在 FastAPI Gateway 中;脚本和 Docker 部署
不会加载本模块。保留它是为了通过 ``langgraph.json`` 的 ``auth.path``
兼容 LangGraph 工具链、Studio 或直接使用 LangGraph Server 的场景。

当使用该兼容路径时,本模块复用与 Gateway 相同的 JWT 和 CSRF 规则,
确保两种模式下的会话校验行为一致。

两层结构:
  1. @auth.authenticate —— 校验 JWT cookie、提取 user_id,
     并对变更状态的方法(POST/PUT/DELETE/PATCH)强制 CSRF 检查
  2. @auth.on —— 返回 metadata 过滤器,使每个用户只能看到自己的 thread
"""

import secrets

from langgraph_sdk import Auth

from app.gateway.auth.errors import TokenError
from app.gateway.auth.jwt import decode_token
from app.gateway.auth_disabled import AUTH_DISABLED_USER_ID, is_auth_disabled
from app.gateway.deps import get_local_provider

auth = Auth()

# 需要 CSRF 校验的方法(按 RFC 7231 属于变更状态的方法)。
_CSRF_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})


def _check_csrf(request) -> None:
    """对变更状态的请求强制执行 Double Submit Cookie CSRF 检查。

    与 Gateway 的 CSRFMiddleware 逻辑保持一致,使由 nginx 直接代理的
    LangGraph 路由享有同等的 CSRF 防护。
    """
    method = getattr(request, "method", "") or ""
    if method.upper() not in _CSRF_METHODS:
        return

    if is_auth_disabled():
        return

    cookie_token = request.cookies.get("csrf_token")
    header_token = request.headers.get("x-csrf-token")

    if not cookie_token or not header_token:
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="CSRF token missing. Include X-CSRF-Token header.",
        )

    if not secrets.compare_digest(cookie_token, header_token):
        raise Auth.exceptions.HTTPException(
            status_code=403,
            detail="CSRF token mismatch.",
        )


@auth.authenticate
async def authenticate(request):
    """校验会话 cookie、解码 JWT 并检查 token_version。

    与 Gateway 的 get_current_user_from_request 相同的校验链:
      cookie → 解码 JWT → 数据库查询 → token_version 匹配
    同时对变更状态的方法强制执行 CSRF 检查。
    """
    # 在认证之前先做 CSRF 检查,即使 cookie 携带有效 JWT,
    # 伪造的跨站请求也会被尽早拒绝。
    _check_csrf(request)

    if is_auth_disabled():
        return AUTH_DISABLED_USER_ID

    token = request.cookies.get("access_token")
    if not token:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Not authenticated",
        )

    payload = decode_token(token)
    if isinstance(payload, TokenError):
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid token",
        )

    user = await get_local_provider().get_user(payload.sub)
    if user is None:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="User not found",
        )
    if user.token_version != payload.ver:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Token revoked (password changed)",
        )

    return payload.sub


@auth.on
async def add_owner_filter(ctx: Auth.types.AuthContext, value: dict):
    """写入时注入 user_id metadata;读取时按 user_id 过滤。

    Gateway 将 thread 归属存储为 ``metadata.user_id``。
    该处理器确保 LangGraph Server 执行相同的隔离。
    """
    # 创建/更新时:将 user_id 写入 metadata
    metadata = value.setdefault("metadata", {})
    metadata["user_id"] = ctx.user.identity

    # 返回过滤字典——LangGraph 将其应用于 search/read/delete
    return {"user_id": ctx.user.identity}
