"""DeerFlow 的授权装饰器与上下文。

灵感来自 LangGraph Auth 系统: https://github.com/langchain-ai/langgraph/blob/main/libs/sdk-py/langgraph_sdk/auth/__init__.py

**用法:**

1. 在需要认证的路由上使用 ``@require_auth``
2. 使用 ``@require_permission("resource", "action", filter_key=...)`` 做权限检查
3. 装饰器链自下而上执行

**示例:**

    @router.get("/{thread_id}")
    @require_auth
    @require_permission("threads", "read", owner_check=True)
    async def get_thread(thread_id: str, request: Request):
        # 用户已认证且拥有 threads:read 权限
        ...

**权限模型:**

- threads:read   - 查看 thread
- threads:write  - 创建/更新 thread
- threads:delete - 删除 thread
- runs:create   - 运行 agent
- runs:read     - 查看 run
- runs:cancel   - 取消 run
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

from fastapi import HTTPException, Request

if TYPE_CHECKING:
    from app.gateway.auth.models import User

P = ParamSpec("P")
T = TypeVar("T")


# 权限常量
class Permissions:
    """resource:action 格式的权限常量。"""

    # Threads
    THREADS_READ = "threads:read"
    THREADS_WRITE = "threads:write"
    THREADS_DELETE = "threads:delete"

    # Runs
    RUNS_CREATE = "runs:create"
    RUNS_READ = "runs:read"
    RUNS_CANCEL = "runs:cancel"


class AuthContext:
    """当前请求的认证上下文。

    经 require_auth 装饰后存储在 request.state.auth 中。

    Attributes:
        user: 已认证用户,匿名时为 None
        permissions: 权限字符串列表(如 "threads:read")
    """

    __slots__ = ("user", "permissions")

    def __init__(self, user: User | None = None, permissions: list[str] | None = None):
        self.user = user
        self.permissions = permissions or []

    @property
    def is_authenticated(self) -> bool:
        """检查用户是否已认证。"""
        return self.user is not None

    def has_permission(self, resource: str, action: str) -> bool:
        """检查上下文是否拥有 resource:action 权限。

        Args:
            resource: 资源名(如 "threads")
            action: 操作名(如 "read")

        Returns:
            有权限返回 True
        """
        permission = f"{resource}:{action}"
        return permission in self.permissions

    def require_user(self) -> User:
        """返回用户,未认证时抛出 401。

        Raises:
            未认证时抛出 HTTPException 401
        """
        if not self.user:
            raise HTTPException(status_code=401, detail="Authentication required")
        return self.user


def get_auth_context(request: Request) -> AuthContext | None:
    """从 request state 获取 AuthContext。"""
    return getattr(request.state, "auth", None)


_ALL_PERMISSIONS: list[str] = [
    Permissions.THREADS_READ,
    Permissions.THREADS_WRITE,
    Permissions.THREADS_DELETE,
    Permissions.RUNS_CREATE,
    Permissions.RUNS_READ,
    Permissions.RUNS_CANCEL,
]


def _make_test_request_stub() -> Any:
    """创建一个供直接单元调用使用的最小化 request 对象。

    用于在没有 FastAPI request 注入时调用被装饰的路由处理器,
    包含认证辅助函数访问的字段。
    """
    return SimpleNamespace(state=SimpleNamespace(), cookies={}, _deerflow_test_bypass_auth=True)


async def _authenticate(request: Request) -> AuthContext:
    """认证请求并返回 AuthContext。

    委托 deps.get_optional_user_from_request() 完成 JWT→User 流程。
    匿名请求返回 user=None 的 AuthContext。
    """
    from app.gateway.deps import get_optional_user_from_request

    user = await get_optional_user_from_request(request)
    if user is None:
        return AuthContext(user=None, permissions=[])

    # 未来可将权限存入用户记录
    return AuthContext(user=user, permissions=_ALL_PERMISSIONS)


def require_auth[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    """对请求进行认证并强制要求已认证的装饰器。

    无论 ASGI 栈中是否存在 ``AuthMiddleware``,对未认证请求独立抛出
    HTTP 401。将解析得到的 ``AuthContext`` 写入 ``request.state.auth``
    供下游处理器使用。

    必须放在其他装饰器之上(即最后执行)。

    Usage:
        @router.get("/{thread_id}")
        @require_auth  # 底层装饰器(权限检查后最先执行)
        @require_permission("threads", "read")
        async def get_thread(thread_id: str, request: Request):
            auth: AuthContext = request.state.auth
            ...

    Raises:
        HTTPException: 请求未认证时返回 401。
        ValueError: 缺少 'request' 参数时抛出。
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        request = kwargs.get("request")
        if request is None:
            # 单元测试可能在没有 FastAPI Request 对象时直接调用被装饰的
            # 处理器。当被包装函数声明了 `request` 参数时,注入一个最小化
            # 的 request 桩对象。
            if "request" in inspect.signature(func).parameters:
                kwargs["request"] = _make_test_request_stub()
            else:
                raise ValueError("require_auth decorator requires 'request' parameter")
            request = kwargs["request"]

        if getattr(request, "_deerflow_test_bypass_auth", False):
            return await func(*args, **kwargs)

        # 认证并设置上下文
        auth_context = await _authenticate(request)
        request.state.auth = auth_context

        if not auth_context.is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")

        return await func(*args, **kwargs)

    return wrapper


def require_permission(
    resource: str,
    action: str,
    owner_check: bool = False,
    require_existing: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """检查 resource:action 权限的装饰器。

    必须在 @require_auth 之后使用。

    Args:
        resource: 资源名(如 "threads"、"runs")
        action: 操作名(如 "read"、"write"、"delete")
        owner_check: 为 True 时校验当前用户是否拥有该资源。
                     需要 'thread_id' 路径参数并执行归属检查。
        require_existing: 仅在 ``owner_check=True`` 时有意义。为 True 时,
                          ``threads_meta`` 行缺失会被视为拒绝(404),而不是
                          "未跟踪的旧 thread,放行"。用于**破坏性 / 变更类**路由
                          (DELETE、PATCH、状态更新),防止已删除的 thread 被其他
                          用户通过"行缺失"代码路径重新定位。

    Usage:
        # 读取类:未跟踪的旧 thread 放行
        @require_permission("threads", "read", owner_check=True)
        async def get_thread(thread_id: str, request: Request):
            ...

        # 破坏类:thread 行必须存在且归调用者所有
        @require_permission("threads", "delete", owner_check=True, require_existing=True)
        async def delete_thread(thread_id: str, request: Request):
            ...

    Raises:
        HTTPException 401: 需要认证但用户为匿名
        HTTPException 403: 用户缺少权限
        HTTPException 404: owner_check=True 但用户不拥有该 thread
        ValueError: owner_check=True 但缺少 'thread_id' 参数
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = kwargs.get("request")
            if request is None:
                # 单元测试可能在没有构造 FastAPI Request 对象时直接调用被
                # 装饰的路由处理器。当被包装函数声明了 `request` 参数时,
                # 注入一个最小化桩对象。
                if "request" in inspect.signature(func).parameters:
                    kwargs["request"] = _make_test_request_stub()
                else:
                    return await func(*args, **kwargs)
                request = kwargs["request"]

            if getattr(request, "_deerflow_test_bypass_auth", False):
                return await func(*args, **kwargs)

            auth: AuthContext = getattr(request.state, "auth", None)
            if auth is None:
                auth = await _authenticate(request)
                request.state.auth = auth

            if not auth.is_authenticated:
                raise HTTPException(status_code=401, detail="Authentication required")

            # 检查权限
            if not auth.has_permission(resource, action):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permission denied: {resource}:{action}",
                )

            # 针对 thread 类资源的归属检查。
            #
            # 2.0-rc 将 thread 元数据迁移到 SQL 持久层(``threads_meta``
            # 表)。我们通过 ``ThreadMetaStore.check_access`` 校验归属:
            # 行缺失(未跟踪的旧 thread)或行的 ``user_id`` 为 NULL(共享 /
            # 认证前的数据)时返回 True,因此这是"严格拒绝"而非"严格放行"
            # ——只有*存在*的行且 user_id *不同*才会触发 404。
            if owner_check:
                from app.gateway.internal_auth import INTERNAL_OWNER_USER_ID_HEADER_NAME, INTERNAL_SYSTEM_ROLE

                thread_id = kwargs.get("thread_id")
                if thread_id is None:
                    raise ValueError("require_permission with owner_check=True requires 'thread_id' parameter")

                from app.gateway.deps import get_thread_store

                thread_store = get_thread_store(request)
                allowed = await thread_store.check_access(
                    thread_id,
                    str(auth.user.id),
                    require_existing=require_existing,
                )
                if not allowed and getattr(auth.user, "system_role", None) == INTERNAL_SYSTEM_ROLE:
                    # 受信任的内部调用方(channel worker)也代表
                    # X-DeerFlow-Owner-User-Id 头中携带的连接属主行事。
                    # 将检查范围限定到该属主,而不是直接放行;泄露的内部
                    # token 绝不能获得跨用户的 thread 访问权。只有在 ``auth``
                    # 已证明调用方持有内部 token 之后才认可该头(与
                    # get_trusted_internal_owner_user_id 一致,后者以中间件
                    # 写入的 ``request.state.user`` 为准)。
                    header_owner = (request.headers.get(INTERNAL_OWNER_USER_ID_HEADER_NAME) or "").strip()
                    if header_owner:
                        allowed = await thread_store.check_access(
                            thread_id,
                            header_owner,
                            require_existing=require_existing,
                        )
                if not allowed:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Thread {thread_id} not found",
                    )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
