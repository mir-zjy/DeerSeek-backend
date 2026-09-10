import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.gateway.auth_disabled import warn_if_auth_disabled_enabled
from app.gateway.auth_middleware import AuthMiddleware
from app.gateway.config import get_gateway_config
from app.gateway.csrf_middleware import CSRFMiddleware, get_configured_cors_origins
from app.gateway.deps import langgraph_runtime
from app.gateway.routers import (
    agents,
    artifacts,
    assistants_compat,
    auth,
    channel_connections,
    channels,
    feedback,
    mcp,
    memory,
    models,
    runs,
    skills,
    suggestions,
    thread_runs,
    threads,
    uploads,
)
from deerflow.config import app_config as deerflow_app_config
from deerflow.config.app_config import apply_logging_level

AppConfig = deerflow_app_config.AppConfig
get_app_config = deerflow_app_config.get_app_config

# 默认日志配置；lifespan 阶段会根据 config.yaml 的 log_level 覆盖。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# 每个 lifespan 关闭钩子允许运行的最长时间（秒）。
# 用于限制 worker 退出时间，避免 uvicorn 的 reload 监控进程
# 不断向卡在关闭清理中的 worker 发送信号。
_SHUTDOWN_HOOK_TIMEOUT_SECONDS = 5.0


async def _ensure_admin_user(app: FastAPI) -> None:
    """启动钩子：处理首次启动，否则迁移孤儿 thread。

    创建管理员后，将 LangGraph store 中的孤儿 thread
    （metadata.user_id 未设置）迁移到管理员账号。这是
    "无认证 → 有认证" 的升级路径：曾在无认证模式下运行
    DeerFlow 的用户已有 LangGraph thread 数据，需要为其
    指定属主。
        首次启动（尚不存在管理员）：
            - 不会自动创建任何用户账号。
            - 运维人员需访问 ``/setup`` 创建第一个管理员。

    后续启动（管理员已存在）：
      - 执行一次性的 "无认证 → 有认证" 孤儿 thread 迁移，
        处理缺少 user_id 的存量 LangGraph thread 元数据。

    无需 SQL 持久化迁移：四个 user_id 列
    （threads_meta、runs、run_events、feedback）只随认证模块
    通过 create_all 一起创建，因此新建的表不会存在属主为
    NULL 的行。
    """
    from sqlalchemy import select

    from app.gateway.deps import get_local_provider
    from deerflow.persistence.engine import get_session_factory
    from deerflow.persistence.user.model import UserRow

    try:
        provider = get_local_provider()
    except RuntimeError:
        # 某些测试/启动路径下认证持久化可能尚未初始化。
        # 跳过管理员迁移工作，而不是让网关启动失败。
        logger.warning("Auth persistence not ready; skipping admin bootstrap check")
        return

    sf = get_session_factory()
    if sf is None:
        return

    admin_count = await provider.count_admin_users()

    if admin_count == 0:
        logger.info("=" * 60)
        logger.info("  检测到首次启动 —— 尚无管理员账号。")
        logger.info("  请访问 /setup 完成管理员账号创建。")
        logger.info("=" * 60)
        return

    # 管理员已存在 —— 对早于认证模块的 LangGraph thread 元数据
    # 执行孤儿 thread 迁移。
    async with sf() as session:
        stmt = select(UserRow).where(UserRow.system_role == "admin").limit(1)
        row = (await session.execute(stmt)).scalar_one_or_none()

    if row is None:
        return  # 理论上不会发生（上面 admin_count > 0），但稳妥起见。

    admin_id = str(row.id)

    # LangGraph store 孤儿迁移 —— 非致命。
    # 覆盖 "无认证 → 有认证" 升级路径：用户的存量 LangGraph
    # thread 元数据未设置 user_id。
    store = getattr(app.state, "store", None)
    if store is not None:
        try:
            migrated = await _migrate_orphaned_threads(store, admin_id)
            if migrated:
                logger.info("已将 %d 个孤儿 LangGraph thread 迁移至管理员账号", migrated)
        except Exception:
            logger.exception("LangGraph thread migration failed (non-fatal)")


async def _iter_store_items(store, namespace, *, page_size: int = 500):
    """对 LangGraph store 命名空间的分页异步迭代器。

    用游标式循环替代旧的硬编码 ``limit=1000`` 调用，避免孤儿数据
    超过一页时静默丢数据。当某页为空或返回不满一页（表示最后一页）
    时终止。
    """
    offset = 0
    while True:
        batch = await store.asearch(namespace, limit=page_size, offset=offset)
        if not batch:
            return
        for item in batch:
            yield item
        if len(batch) < page_size:
            return
        offset += page_size


async def _migrate_orphaned_threads(store, admin_user_id: str) -> int:
    """将 LangGraph store 中没有 user_id 的 thread 迁移给指定管理员。

    使用游标分页，无论孤儿数量多少都能全部迁移。返回迁移的行数。
    """
    migrated = 0
    async for item in _iter_store_items(store, ("threads",)):
        metadata = item.value.get("metadata", {})
        if not metadata.get("user_id"):
            metadata["user_id"] = admin_user_id
            item.value["metadata"] = metadata
            await store.aput(("threads",), item.key, item.value)
            migrated += 1
    return migrated


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期处理器。"""

    # 启动时加载配置并检查必要的环境变量。
    # `startup_config` 是本地快照，仅用于一次性的引导工作
    # （日志级别、langgraph_runtime 引擎、channels）。请求期的配置
    # 解析始终通过 `app/gateway/deps.py::get_config()` 中的
    # `get_app_config()` 进行，因此对 `config.yaml` 的修改无需重启
    # 进程即可生效。我们特意不将该快照缓存到 `app.state` 上，
    # 以保证这一约定可被强制执行。
    try:
        startup_config = get_app_config()
        apply_logging_level(startup_config.log_level)
        logger.info("配置加载成功")
        warn_if_auth_disabled_enabled()
    except Exception as e:
        error_msg = f"Failed to load configuration during gateway startup: {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e
    config = get_gateway_config()
    logger.info(f"API 网关启动中，监听 {config.host}:{config.port}")

    # 预热 tiktoken 编码缓存，使首个注入记忆的请求不会因下载 BPE
    # 数据而阻塞（该下载会访问 OpenAI/Azure 的 URL，在受限网络中可能
    # 不可达 —— 见 issue #3402）。
    # 当 memory.token_counting 为 "char" 时，token 计数完全不涉及
    # tiktoken，因此直接跳过预热（在网络受限的部署中连 5 秒探测也省掉
    # —— 见 issue #3429）。
    if startup_config.memory.token_counting == "char":
        logger.info("memory.token_counting='char'，跳过 tiktoken 预热（使用无网络的字符估算）")
    else:
        try:
            from deerflow.agents.memory.prompt import warm_tiktoken_cache

            warmed = await asyncio.wait_for(
                asyncio.to_thread(warm_tiktoken_cache),
                timeout=5,
            )
            if warmed:
                logger.info("tiktoken 编码缓存预热成功")
            else:
                logger.warning("tiktoken encoding cache warm-up failed; token counting will use character-based fallback until tiktoken loads successfully")
        except TimeoutError:
            logger.warning("tiktoken encoding cache warm-up timed out; token counting will use character-based fallback until tiktoken loads successfully")
        except Exception:
            logger.warning("tiktoken warm-up skipped", exc_info=True)

    # 初始化 LangGraph 运行时组件（StreamBridge、RunManager、checkpointer、store）
    async with langgraph_runtime(app, startup_config):
        logger.info("LangGraph 运行时初始化完成")

        # 检查管理员引导状态，并在管理员存在后迁移孤儿 thread。
        # 必须在 langgraph_runtime 之后运行，此时 app.state.store 才可用于 thread 迁移
        await _ensure_admin_user(app)

        # 如果配置了任何 channel，则启动 IM channel 服务
        try:
            from app.channels.service import start_channel_service

            channel_service = await start_channel_service(startup_config)
            logger.info("Channel 服务已启动: %s", channel_service.get_status())
        except Exception:
            logger.exception("No IM channels configured or channel service failed to start")

        yield

        # 关闭时停止 channel 服务（设置时限，防止 worker 挂起）
        try:
            from app.channels.service import stop_channel_service

            await asyncio.wait_for(
                stop_channel_service(),
                timeout=_SHUTDOWN_HOOK_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Channel service shutdown exceeded %.1fs; proceeding with worker exit.",
                _SHUTDOWN_HOOK_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Failed to stop channel service")

    logger.info("API 网关正在关闭")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。

    Returns:
        配置好的 FastAPI 应用实例。
    """
    config = get_gateway_config()
    docs_url = "/docs" if config.enable_docs else None
    redoc_url = "/redoc" if config.enable_docs else None
    openapi_url = "/openapi.json" if config.enable_docs else None

    app = FastAPI(
        title="DeerFlow API Gateway",
        description="""
## DeerFlow API 网关

DeerFlow 的 API 网关——基于 LangGraph 的 AI 智能体后端，具备沙盒执行能力。

### 功能特性

- **模型管理**：查询和获取可用的 AI 模型
- **MCP 配置**：管理模型上下文协议（MCP）服务器配置
- **记忆管理**：访问和管理全局记忆数据，实现个性化对话
- **技能管理**：查询和管理技能及其启用状态
- **工件（Artifacts）**：访问线程工件及生成的文件
- **健康监控**：系统健康检查端点

### 架构

兼容 LangGraph 的请求通过 nginx 路由至此网关。该网关提供用于智能体运行时的端点，以及用于模型、MCP 配置、技能和工件的自定义端点。
        """,
        version="0.1.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        openapi_tags=[
            {
                "name": "models",
                "description": "Operations for querying available AI models and their configurations",
            },
            {
                "name": "mcp",
                "description": "Manage Model Context Protocol (MCP) server configurations",
            },
            {
                "name": "memory",
                "description": "Access and manage global memory data for personalized conversations",
            },
            {
                "name": "skills",
                "description": "Manage skills and their configurations",
            },
            {
                "name": "artifacts",
                "description": "Access and download thread artifacts and generated files",
            },
            {
                "name": "uploads",
                "description": "Upload and manage user files for threads",
            },
            {
                "name": "threads",
                "description": "Manage DeerFlow thread-local filesystem data",
            },
            {
                "name": "agents",
                "description": "Create and manage custom agents with per-agent config and prompts",
            },
            {
                "name": "suggestions",
                "description": "Generate follow-up question suggestions for conversations",
            },
            {
                "name": "channels",
                "description": "Manage IM channel integrations (Feishu, Slack, Telegram)",
            },
            {
                "name": "assistants-compat",
                "description": "LangGraph Platform-compatible assistants API (stub)",
            },
            {
                "name": "runs",
                "description": "LangGraph Platform-compatible runs lifecycle (create, stream, cancel)",
            },
            {
                "name": "health",
                "description": "Health check and system status endpoints",
            },
        ],
    )

    # 认证：拒绝对非公开路径的未认证请求（fail-closed 安全兜底）
    app.add_middleware(AuthMiddleware)

    # CSRF：对会改变状态的请求采用 Double Submit Cookie 模式
    app.add_middleware(CSRFMiddleware)

    # CORS：统一的 nginx 入口默认同源。跨域的浏览器客户端必须通过
    # 这个显式的网关白名单启用，使 CORS 与 CSRF 的来源检查共享同一
    # 事实来源。
    cors_origins = sorted(get_configured_cors_origins())
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 挂载各路由
    # Models API 挂载在 /api/models
    app.include_router(models.router)

    # MCP API 挂载在 /api/mcp
    app.include_router(mcp.router)

    # Memory API 挂载在 /api/memory
    app.include_router(memory.router)

    # Skills API 挂载在 /api/skills
    app.include_router(skills.router)

    # Artifacts API 挂载在 /api/threads/{thread_id}/artifacts
    app.include_router(artifacts.router)

    # Uploads API 挂载在 /api/threads/{thread_id}/uploads
    app.include_router(uploads.router)

    # Thread 清理 API 挂载在 /api/threads/{thread_id}
    app.include_router(threads.router)

    # Agents API 挂载在 /api/agents
    app.include_router(agents.router)

    # Suggestions API 挂载在 /api/threads/{thread_id}/suggestions
    app.include_router(suggestions.router)

    # 面向用户的 IM channel 连接 API 挂载在 /api/channels
    app.include_router(channel_connections.router)

    # Channels API 挂载在 /api/channels
    app.include_router(channels.router)

    # Assistants 兼容 API（LangGraph Platform 桩实现）
    app.include_router(assistants_compat.router)

    # Auth API 挂载在 /api/v1/auth
    app.include_router(auth.router)

    # Feedback API 挂载在 /api/threads/{thread_id}/runs/{run_id}/feedback
    app.include_router(feedback.router)

    # Thread Runs API（兼容 LangGraph Platform 的 run 生命周期）
    app.include_router(thread_runs.router)

    # 无状态 Runs API（无需预先存在 thread 的 stream/wait）
    app.include_router(runs.router)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """健康检查端点。

        Returns:
            服务健康状态信息。
        """
        return {"status": "healthy", "service": "deer-flow-gateway"}

    return app


# 为 uvicorn 创建应用实例
app = create_app()
