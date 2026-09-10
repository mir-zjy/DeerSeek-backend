"""集中管理存放在 ``app.state`` 上的单例对象的访问器。

**Getter**（供路由使用）：必需的依赖缺失时抛出 503，``get_store``
例外，返回 ``None``。

``AppConfig`` 有意*不*缓存到 ``app.state`` 上。路由和 run 路径通过
:func:`deerflow.config.app_config.get_app_config` 解析它，该函数基于
mtime 热重载，因此对 ``config.yaml`` 的修改无需重启进程即可在下一次
请求生效。:func:`langgraph_runtime` 中创建的引擎（stream bridge、
持久化、checkpointer、store、run 事件存储）接收 ``startup_config``
快照 —— 它们在设计上要求重启才变更，并保持绑定到该快照，以保证
运行中的进程内部一致。

初始化在 ``app.py`` 中通过 :class:`AsyncExitStack` 直接处理。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, TypeVar, cast

from fastapi import FastAPI, HTTPException, Request
from langgraph.types import Checkpointer

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.persistence.feedback import FeedbackRepository
from deerflow.runtime import RunContext, RunManager, StreamBridge
from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.runs.store.base import RunStore

logger = logging.getLogger(__name__)

# 关闭过程中排空在途 run 的最长时间（秒），之后 AsyncExitStack 才会拆除
# checkpointer（及其连接池）。保留在本地以避免 app -> deps -> app 的
# 循环导入。这是与 ``app.gateway.app._SHUTDOWN_HOOK_TIMEOUT_SECONDS``
# （目前同样为 5.0 秒，用于限制 channel 服务停止）相互*独立*的预算：
# 两者分别约束独立的拆除步骤，可以不同，但都计入 lifespan 关闭窗口
# —— 如果它们的总和必须落在服务器的优雅关闭超时内，请一并重新审视。
_RUN_DRAIN_TIMEOUT_SECONDS = 5.0


async def _drain_inflight_runs(run_manager: RunManager) -> None:
    """在 checkpointer 被拆除前排空在途 run（issue #3373）。

    对（内部已限时的）排空操作加 shield，即使 lifespan 协程本身在关闭
    中途被取消 —— 第二次 SIGINT 或服务器的优雅关闭超时，即 #3373 背后
    同样的信号风暴 —— checkpointer 连接池也不会在 run 任务仍在写
    checkpoint 时被关闭。发生这种取消时，我们让已在运行的排空完成
    （它有 ``RunManager.shutdown`` 自身的超时约束），然后继续传播取消。
    """
    drain = asyncio.create_task(run_manager.shutdown(timeout=_RUN_DRAIN_TIMEOUT_SECONDS))
    try:
        await asyncio.shield(drain)
    except asyncio.CancelledError:
        # 再次 shield，使第二次等待不会抛弃在途的排空操作；排空已限时，
        # 因此不会挂起。然后重新抛出以响应关闭。
        try:
            await asyncio.shield(drain)
        except Exception:
            logger.exception("In-flight run drain failed after shutdown cancellation")
        raise
    except Exception:
        logger.exception("Failed to drain in-flight runs during shutdown")


if TYPE_CHECKING:
    from app.gateway.auth.local_provider import LocalAuthProvider
    from app.gateway.auth.repositories.sqlite import SQLiteUserRepository
    from deerflow.persistence.thread_meta.base import ThreadMetaStore
    from deerflow.runtime import RunRecord


T = TypeVar("T")


async def _mark_latest_recovered_threads_error(
    run_manager: RunManager,
    thread_store: ThreadMetaStore,
    recovered_runs: list[RunRecord],
) -> None:
    """仅当 thread 最新的 run 被恢复时，才将其状态标记为 error。"""
    recovered_by_thread: dict[str, set[str]] = {}
    for record in recovered_runs:
        recovered_by_thread.setdefault(record.thread_id, set()).add(record.run_id)

    for thread_id, recovered_run_ids in recovered_by_thread.items():
        try:
            latest_runs = await run_manager.list_by_thread(thread_id, user_id=None, limit=1)
        except Exception:
            logger.warning("Failed to find latest run for thread %s during run reconciliation", thread_id, exc_info=True)
            continue
        if not latest_runs or latest_runs[0].run_id not in recovered_run_ids:
            continue
        try:
            await thread_store.update_status(thread_id, "error", user_id=None)
        except Exception:
            logger.warning("Failed to mark thread %s as error during run reconciliation", thread_id, exc_info=True)


def get_config() -> AppConfig:
    """为当前请求返回最新的 ``AppConfig``。

    经由 :func:`deerflow.config.app_config.get_app_config` 获取，它会
    尊重运行时的 ``ContextVar`` 覆盖，并在 ``config.yaml`` 的 mtime 变化
    时从磁盘重载。``AppConfig`` 完全不缓存在 ``app.state`` 上 —— 唯一的
    启动期快照是 ``lifespan()`` 内部的局部变量 ``startup_config``，它被
    显式传入 :func:`langgraph_runtime`，供那些设计上要求重启才变更的
    引擎使用。所有请求都经由 :func:`get_app_config` 路由，修复了
    bytedance/deer-flow issue #3107 BUG-001 中 worker / lead-agent 线程
    看到过期的启动快照的脑裂问题。

    热重载边界：由启动期单例支撑的字段（引擎、sandbox provider、
    IM channel、日志 handler）需要重启进程才能在运行时变更。权威清单
    位于 :mod:`deerflow.config.reload_boundary`，并由 :class:`AppConfig`
    中对应 ``Field(description=...)`` 上标准化的 ``"startup-only:"`` 前缀
    镜像 —— 在 IDE 中悬停这些字段即可内联看到边界说明。运维摘要见
    ``backend/CLAUDE.md`` 的 "Config Hot-Reload Boundary" 一节。

    任何物化配置的失败（文件缺失、权限不足、YAML 解析错误、校验错误）
    都会以 503 报告 —— 语义上是 "网关在缺少可用配置时无法服务请求"
    —— 并连同原始异常记录日志，便于运维排查。
    """
    try:
        return get_app_config()
    except Exception as exc:  # noqa: BLE001 - request boundary: log and degrade gracefully
        logger.exception("Failed to load AppConfig at request time")
        raise HTTPException(status_code=503, detail="Configuration not available") from exc


@asynccontextmanager
async def langgraph_runtime(app: FastAPI, startup_config: AppConfig) -> AsyncGenerator[None, None]:
    """引导并拆除所有 LangGraph 运行时单例。

    ``startup_config`` 是 ``lifespan()`` 中为一次性基础设施引导获取的
    ``AppConfig`` 快照。此处构建的引擎和存储（stream bridge、持久化引擎、
    checkpointer、store、run 事件存储）在设计上要求重启才变更 —— 它们
    持有活跃连接、文件句柄或单例 provider —— 因此绑定到该快照，并在
    `config.yaml` 修改后继续存活。请求期消费方对于应当热重载的字段
    仍须经由 :func:`get_config` 获取。见 ``backend/CLAUDE.md`` 的
    "Config Hot-Reload Boundary" 一节。

    对应的 ``run_events_config`` 被冻结到 ``app.state`` 上，使
    :func:`get_run_context` 把新加载的 ``AppConfig`` 与底层
    ``event_store`` 构建时所用的*启动期* run 事件配置配对 —— 否则
    运行时可能把新的 ``run_events_config`` 与仍绑定旧后端的
    event store 组合在一起。

    在 ``app.py`` 中的用法::

        async with langgraph_runtime(app, startup_config):
            yield
    """
    from deerflow.persistence.engine import close_engine, get_session_factory, init_engine_from_config
    from deerflow.runtime import make_store, make_stream_bridge
    from deerflow.runtime.checkpointer.async_provider import make_checkpointer
    from deerflow.runtime.events.store import make_run_event_store

    async with AsyncExitStack() as stack:
        config = startup_config

        app.state.stream_bridge = await stack.enter_async_context(make_stream_bridge(config))

        # 在 checkpointer 之前初始化持久化引擎，使自动建库逻辑先执行
        # （postgres 后端）。
        await init_engine_from_config(config.database)

        app.state.checkpointer = await stack.enter_async_context(make_checkpointer(config))
        app.state.store = await stack.enter_async_context(make_store(config))

        # 初始化各 repository —— 所有 repository 共用一次 get_session_factory() 调用。
        sf = get_session_factory()
        if sf is not None:
            from deerflow.persistence.feedback import FeedbackRepository
            from deerflow.persistence.run import RunRepository

            app.state.run_store = RunRepository(sf)
            app.state.feedback_repo = FeedbackRepository(sf)
        else:
            from deerflow.runtime.runs.store.memory import MemoryRunStore

            app.state.run_store = MemoryRunStore()
            app.state.feedback_repo = None

        from deerflow.persistence.thread_meta import make_thread_store

        app.state.thread_store = make_thread_store(sf, app.state.store)

        # Run 事件存储。store 和对应的 ``run_events_config`` 都在启动时
        # 冻结，避免 ``get_run_context`` 把刚重载的 ``AppConfig.run_events``
        # 与仍绑定旧后端的 store 组合在一起。
        run_events_config = getattr(config, "run_events", None)
        app.state.run_events_config = run_events_config
        app.state.run_event_store = make_run_event_store(run_events_config)

        # 带持久化 store 的 RunManager
        app.state.run_manager = RunManager(store=app.state.run_store)
        if getattr(config.database, "backend", None) == "sqlite":
            from deerflow.utils.time import now_iso

            # 仅启动时执行的恢复：正常关闭时不会有活跃行返回，下面的
            # thread 状态更新即为无操作。
            recovered_runs = await app.state.run_manager.reconcile_orphaned_inflight_runs(
                error="Gateway restarted before this run reached a durable final state.",
                before=now_iso(),
            )
            await _mark_latest_recovered_threads_error(app.state.run_manager, app.state.thread_store, recovered_runs)

        try:
            yield
        finally:
            # 在 AsyncExitStack 拆除 checkpointer（及其连接池）*之前*排空
            # 在途 run 任务。仍在图执行中途的 run 否则会泄漏进 asyncio.run()
            # 的关闭阶段，langgraph 的 _checkpointer_put_after_previous 的
            # aput 会与已关闭的连接池竞争并抛出 PoolClosed（issue #3373）。
            run_manager = getattr(app.state, "run_manager", None)
            if run_manager is not None:
                await _drain_inflight_runs(run_manager)
            await close_engine()


# ---------------------------------------------------------------------------
# Getter —— 由路由按请求调用
# ---------------------------------------------------------------------------


def _require(attr: str, label: str) -> Callable[[Request], T]:
    """创建一个 FastAPI 依赖：返回 ``app.state.<attr>``，缺失时抛 503。"""

    def dep(request: Request) -> T:
        val = getattr(request.app.state, attr, None)
        if val is None:
            raise HTTPException(status_code=503, detail=f"{label} not available")
        return cast(T, val)

    dep.__name__ = dep.__qualname__ = f"get_{attr}"
    return dep


get_stream_bridge: Callable[[Request], StreamBridge] = _require("stream_bridge", "Stream bridge")
get_run_manager: Callable[[Request], RunManager] = _require("run_manager", "Run manager")
get_checkpointer: Callable[[Request], Checkpointer] = _require("checkpointer", "Checkpointer")
get_run_event_store: Callable[[Request], RunEventStore] = _require("run_event_store", "Run event store")
get_feedback_repo: Callable[[Request], FeedbackRepository] = _require("feedback_repo", "Feedback")
get_run_store: Callable[[Request], RunStore] = _require("run_store", "Run store")


def get_store(request: Request):
    """返回全局 store（未配置时可能为 ``None``）。"""
    return getattr(request.app.state, "store", None)


def get_thread_store(request: Request) -> ThreadMetaStore:
    """返回 thread 元数据存储（SQL 或内存实现）。"""
    val = getattr(request.app.state, "thread_store", None)
    if val is None:
        raise HTTPException(status_code=503, detail="Thread metadata store not available")
    return val


def get_run_context(request: Request) -> RunContext:
    """用 ``app.state`` 单例构建 :class:`RunContext`。

    返回带基础设施依赖的*基础*上下文。``app_config`` 字段实时解析，
    使按 run 生效的字段（如 ``models[*].max_tokens``）跟随
    ``config.yaml`` 的修改；``event_store`` / ``run_events_config``
    这对组合保持冻结在 :func:`langgraph_runtime` 捕获的快照上，使调用方
    永远不会看到绑定一个后端的 store 搭配指向另一个后端的配置。
    """
    return RunContext(
        checkpointer=get_checkpointer(request),
        store=get_store(request),
        event_store=get_run_event_store(request),
        run_events_config=getattr(request.app.state, "run_events_config", None),
        thread_store=get_thread_store(request),
        app_config=get_config(),
    )


# ---------------------------------------------------------------------------
# 认证辅助函数（供 authz.py 和认证中间件使用）
# ---------------------------------------------------------------------------

# 缓存的单例，避免每次请求重复实例化
_cached_local_provider: LocalAuthProvider | None = None
_cached_repo: SQLiteUserRepository | None = None


def get_local_provider() -> LocalAuthProvider:
    """获取或创建缓存的 LocalAuthProvider 单例。

    必须在 ``init_engine_from_config()`` 之后调用 —— 构建用户 repository
    需要共享的 session factory。
    """
    global _cached_local_provider, _cached_repo
    if _cached_repo is None:
        from app.gateway.auth.repositories.sqlite import SQLiteUserRepository
        from deerflow.persistence.engine import get_session_factory

        sf = get_session_factory()
        if sf is None:
            raise RuntimeError("get_local_provider() called before init_engine_from_config(); cannot access users table")
        _cached_repo = SQLiteUserRepository(sf)
    if _cached_local_provider is None:
        from app.gateway.auth.local_provider import LocalAuthProvider

        _cached_local_provider = LocalAuthProvider(repository=_cached_repo)
    return _cached_local_provider


async def get_current_user_from_request(request: Request):
    """从请求 cookie 获取当前已认证用户。

    未认证时抛出 HTTPException 401。
    """
    state = getattr(request, "state", None)
    state_user = getattr(state, "user", None)
    from app.gateway.auth_disabled import AUTH_SOURCE_AUTH_DISABLED, AUTH_SOURCE_INTERNAL, AUTH_SOURCE_SESSION

    if state_user is not None and getattr(state, "auth_source", None) in {
        AUTH_SOURCE_SESSION,
        AUTH_SOURCE_AUTH_DISABLED,
        AUTH_SOURCE_INTERNAL,
    }:
        return state_user

    from app.gateway.auth import decode_token
    from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse, TokenError, token_error_to_code

    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=AuthErrorCode.NOT_AUTHENTICATED, message="Not authenticated").model_dump(),
        )

    payload = decode_token(access_token)
    if isinstance(payload, TokenError):
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=token_error_to_code(payload), message=f"Token error: {payload.value}").model_dump(),
        )

    provider = get_local_provider()
    user = await provider.get_user(payload.sub)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=AuthErrorCode.USER_NOT_FOUND, message="User not found").model_dump(),
        )

    # token 版本不匹配 → 密码已修改，token 已失效
    if user.token_version != payload.ver:
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=AuthErrorCode.TOKEN_INVALID, message="Token revoked (password changed)").model_dump(),
        )

    return user


async def require_admin_user(request: Request, *, detail: str) -> None:
    """要求已认证的调用方必须是管理员用户。

    ``AuthMiddleware`` 通常会在请求到达路由前把 ``request.state.user``
    盖戳。回退到严格依赖可在测试或其他不挂载全局中间件的 ASGI 组合中
    保持路由安全。``detail`` 是该路由专属的 403 消息。

    把此逻辑集中在这里，意味着将来对管理员定义的修改（如允许内部系统
    角色、增加审计日志、或切换为基于权限的检查）只需落地一处，而不是在
    之前分散于 ``mcp``、``channel_connections`` 和 ``channels`` 的各路由
    副本间漂移。
    """
    user = getattr(request.state, "user", None)
    if user is None:
        user = await get_current_user_from_request(request)

    if getattr(user, "system_role", None) != "admin":
        raise HTTPException(status_code=403, detail=detail)


async def get_optional_user_from_request(request: Request):
    """从请求获取可选的已认证用户。

    未认证时返回 None。
    """
    try:
        return await get_current_user_from_request(request)
    except HTTPException:
        return None


async def get_current_user(request: Request) -> str | None:
    """从请求 cookie 提取 user_id，未认证时返回 None。

    为只需要身份标识的调用方（如 ``feedback.py``）返回字符串 id 的薄
    适配器。需要完整用户对象的调用方应使用
    ``get_current_user_from_request`` 或 ``get_optional_user_from_request``。
    """
    user = await get_optional_user_from_request(request)
    return str(user.id) if user else None
