"""Run 生命周期服务层。

集中实现创建 run、格式化 SSE 帧、消费 stream bridge 事件等业务逻辑。
路由模块（``thread_runs``、``runs``）只是转发到此处的薄 HTTP 处理器。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException, Request
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_messages

from app.gateway.deps import get_run_context, get_run_manager, get_stream_bridge
from app.gateway.internal_auth import INTERNAL_SYSTEM_ROLE, get_trusted_internal_owner_user_id
from app.gateway.utils import sanitize_log_param
from deerflow.config.app_config import get_app_config
from deerflow.runtime import (
    END_SENTINEL,
    HEARTBEAT_SENTINEL,
    ConflictError,
    DisconnectMode,
    RunManager,
    RunRecord,
    RunStatus,
    StreamBridge,
    UnsupportedStrategyError,
    run_agent,
)
from deerflow.runtime.runs.naming import resolve_root_run_name
from deerflow.runtime.user_context import reset_current_user, set_current_user

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSE 格式化
# ---------------------------------------------------------------------------


def format_sse(event: str, data: Any, *, event_id: str | None = None) -> str:
    """格式化单个 SSE 帧。

    字段顺序：``event:`` -> ``data:`` -> ``id:``（可选）-> 空行。
    与 LangGraph Platform 的线上格式一致，可被 ``useStream`` React
    hook 和 Python ``langgraph-sdk`` 的 SSE 解码器消费。
    """
    payload = json.dumps(data, default=str, ensure_ascii=False)
    parts = [f"event: {event}", f"data: {payload}"]
    if event_id:
        parts.append(f"id: {event_id}")
    parts.append("")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 输入 / 配置辅助函数
# ---------------------------------------------------------------------------


def normalize_stream_modes(raw: list[str] | str | None) -> list[str]:
    """将 stream_mode 参数归一化为列表。

    默认值与 ``useStream`` 的期望一致：values + messages-tuple。
    """
    if raw is None:
        return ["values"]
    if isinstance(raw, str):
        return [raw]
    return raw if raw else ["values"]


def normalize_input(raw_input: dict[str, Any] | None) -> dict[str, Any]:
    """将 LangGraph Platform 的输入格式转换为 LangChain 状态字典。

    字典→消息的转换委托给 ``langchain_core.messages.utils.convert_to_messages``，
    使 ``additional_kwargs``（如上传文件的元数据 —— gh #3132）、``id``、
    ``name`` 以及非 human 角色（ai/system/tool）原样保留。早期手写版本
    只转发 ``content`` 且把所有角色都折叠为 ``HumanMessage``，会静默丢掉
    前端提供的附件。

    格式错误的消息字典（缺少 ``role``/``type``/``content``、不支持的角色
    等）会抛出带出错下标的 ``HTTPException(400)``，而不是冒泡成 500。
    网关是系统边界，逐条返回校验错误更便于客户端重试。
    """
    if raw_input is None:
        return {}
    messages = raw_input.get("messages")
    if messages and isinstance(messages, list):
        converted: list[Any] = []
        for index, msg in enumerate(messages):
            if isinstance(msg, BaseMessage):
                converted.append(msg)
            elif isinstance(msg, dict):
                try:
                    converted.extend(convert_to_messages([msg]))
                except (ValueError, TypeError, NotImplementedError) as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid message at input.messages[{index}]: {exc}",
                    ) from exc
            else:
                converted.append(msg)
        return {**raw_input, "messages": converted}
    return raw_input


_DEFAULT_ASSISTANT_ID = "lead_agent"


# langgraph-compat 层从 ``body.context`` 转发到 run 配置的 run 上下文键
# 白名单。LangGraph >=0.6 引入了 ``config["context"]``，但这些值必须同时
# 写入 ``configurable``（供旧的 ``_get_runtime_config`` 消费方使用）和
# ``context``，因为 LangGraph >=1.1.9 中 ``ToolRuntime.context`` 不再回退到
# ``configurable`` 供 ``setup_agent`` 等消费方读取。
_CONTEXT_CONFIGURABLE_KEYS: frozenset[str] = frozenset(
    {
        "model_name",
        "mode",
        "thinking_enabled",
        "reasoning_effort",
        "is_plan_mode",
        "subagent_enabled",
        "max_concurrent_subagents",
        "agent_name",
        "is_bootstrap",
    }
)


def merge_run_context_overrides(config: dict[str, Any], context: Mapping[str, Any] | None) -> None:
    """将 ``body.context`` 中的白名单键合并到 ``config['configurable']``
    和 ``config['context']`` 两处，使其对旧的 configurable 读取方和
    LangGraph ``ToolRuntime.context`` 消费方（如 ``setup_agent`` 工具 ——
    见 issue #2677）都可见。

    ``user_id`` 除白名单键之外也被有意传播到 ``config['context']``，使
    在 ``body.context`` 中携带身份的非 Web 调用方（如 IM channel）能在
    ``ToolRuntime.context`` 上保留它。合并使用 ``setdefault``，因此由
    :func:`inject_authenticated_user_context` 盖上的服务端认证 id 始终
    优先于客户端提供的值。
    """
    if not context:
        return
    configurable = config.setdefault("configurable", {})
    runtime_context = config.setdefault("context", {})
    for key in _CONTEXT_CONFIGURABLE_KEYS:
        if key in context:
            if isinstance(configurable, dict):
                configurable.setdefault(key, context[key])
            if isinstance(runtime_context, dict):
                runtime_context.setdefault(key, context[key])
    if "user_id" in context and isinstance(runtime_context, dict):
        runtime_context.setdefault("user_id", context["user_id"])


def inject_authenticated_user_context(config: dict[str, Any], request: Request) -> None:
    """将已认证用户盖进 run 上下文，供后台工具使用。

    工具可能在请求处理器返回之后才执行，因此需要持久化用户级文件的工具
    不应只依赖周围的 ContextVar。该值来自服务端认证状态，绝不来自客户端
    上下文。
    """

    user = getattr(request.state, "user", None)
    user_id = getattr(user, "id", None)
    if user_id is None:
        return

    if getattr(user, "system_role", None) == INTERNAL_SYSTEM_ROLE:
        return

    runtime_context = config.setdefault("context", {})
    if isinstance(runtime_context, dict):
        runtime_context["user_id"] = str(user_id)


def resolve_agent_factory(assistant_id: str | None):
    """从配置解析 agent 工厂可调用对象。

    自定义 agent 以 ``lead_agent`` + 注入到 ``configurable`` 或
    ``context`` 的 ``agent_name`` 实现 —— 见 :func:`build_run_config`。
    因此所有 ``assistant_id`` 都映射到同一个工厂；真正的路由发生在
    ``make_lead_agent`` 读取 ``cfg["agent_name"]`` 时。
    """
    from deerflow.agents.lead_agent.agent import make_lead_agent

    return make_lead_agent


def build_run_config(
    thread_id: str,
    request_config: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
    *,
    assistant_id: str | None = None,
) -> dict[str, Any]:
    """为 agent 构建 RunnableConfig 字典。

    当 *assistant_id* 指向自定义 agent（即除 ``"lead_agent"`` / ``None``
    以外的值）时，其名称会作为 ``agent_name`` 同时转发到 ``configurable``
    和 ``context``，使其对旧的 configurable 读取方和 LangGraph
    ``ToolRuntime.context`` 消费方（如 ``setup_agent`` 工具，自
    LangGraph >=1.1.9 起不再从 ``context`` 回退到 ``configurable``）都可见。
    任一容器中显式提供的 ``agent_name`` 优先于由 ``assistant_id`` 派生的值。
    ``make_lead_agent`` 读取该键来加载对应的 ``agents/<name>/SOUL.md``
    和每个 agent 各自的配置 —— 没有它，agent 会静默地以默认 lead agent
    身份运行。

    这与 channel 管理器的 ``_resolve_run_params`` 逻辑保持一致，使
    兼容 LangGraph Platform 的 HTTP API 与 IM channel 路径行为完全相同。
    """
    # lead agent 的递归预算（仅针对 lead 图的 LangGraph 超步）。
    # 与子 agent 深度无关：一次 `task()` 派发会在 lead 的单个工具节点步骤
    # 内运行整个子 agent，而子 agent 通过 `subagents.max_turns` 自行限制。
    # 不要把这里的 100 与通用子 agent 的 max_turns 混为一谈。
    config: dict[str, Any] = {"recursion_limit": 100}
    if request_config:
        # LangGraph >= 0.6.0 引入 ``context`` 作为传递线程级数据的推荐
        # 方式，并拒绝同时包含 ``configurable`` 和 ``context`` 的请求。
        # 如果调用方已发送 ``context``，则尊重它并跳过我们自己的
        # ``configurable`` 字典。
        if "context" in request_config:
            if "configurable" in request_config:
                logger.warning(
                    "build_run_config: client sent both 'context' and 'configurable'; preferring 'context' (LangGraph >= 0.6.0). thread_id=%s, caller_configurable keys=%s",
                    thread_id,
                    list(request_config.get("configurable", {}).keys()),
                )
            context_value = request_config["context"]
            if context_value is None:
                context = {}
            elif isinstance(context_value, Mapping):
                context = dict(context_value)
            else:
                raise ValueError("request config 'context' must be a mapping or null.")
            config["context"] = context
        else:
            configurable = {"thread_id": thread_id}
            configurable.update(request_config.get("configurable", {}))
            config["configurable"] = configurable
        for k, v in request_config.items():
            if k not in ("configurable", "context"):
                config[k] = v
    else:
        config["configurable"] = {"thread_id": thread_id}

    # 当调用方指定了非默认 assistant 时，注入自定义 agent 名称。
    # 任一运行时选项容器中显式的 agent_name 都会被尊重。
    if assistant_id and assistant_id != _DEFAULT_ASSISTANT_ID:
        normalized = assistant_id.strip().lower().replace("_", "-")
        if not normalized or not re.fullmatch(r"[a-z0-9-]+", normalized):
            raise ValueError(f"Invalid assistant_id {assistant_id!r}: must contain only letters, digits, and hyphens after normalization.")
        configurable = config.setdefault("configurable", {})
        runtime_context = config.setdefault("context", {})
        explicit_agent_name: str | None = None
        if isinstance(configurable, dict) and isinstance(configurable.get("agent_name"), str):
            explicit_agent_name = configurable["agent_name"]
        elif isinstance(runtime_context, dict) and isinstance(runtime_context.get("agent_name"), str):
            explicit_agent_name = runtime_context["agent_name"]
        effective_agent_name = explicit_agent_name or normalized
        if isinstance(configurable, dict):
            configurable["agent_name"] = effective_agent_name
        if isinstance(runtime_context, dict):
            runtime_context["agent_name"] = effective_agent_name
        config.setdefault("run_name", resolve_root_run_name(config, normalized))
    if metadata:
        config.setdefault("metadata", {}).update(metadata)
    return config


# ---------------------------------------------------------------------------
# Run 生命周期
# ---------------------------------------------------------------------------


async def start_run(
    body: Any,
    thread_id: str,
    request: Request,
) -> RunRecord:
    """创建 RunRecord 并启动后台 agent 任务。

    Parameters
    ----------
    body : RunCreateRequest
        已校验的请求体（类型标注为 Any，以避免与定义 Pydantic 模型的
        路由模块产生循环导入）。
    thread_id : str
        目标 thread。
    request : Request
        FastAPI 请求 —— 用于从 ``app.state`` 获取单例。
    """
    bridge = get_stream_bridge(request)
    run_mgr = get_run_manager(request)
    run_ctx = get_run_context(request)

    disconnect = DisconnectMode.cancel if body.on_disconnect == "cancel" else DisconnectMode.continue_

    body_context = getattr(body, "context", None) or {}
    model_name = body_context.get("model_name")

    # 截断前先将非字符串类型的 model_name 强制转换为 str。
    if model_name is not None and not isinstance(model_name, str):
        model_name = str(model_name)

    # 提供了 model_name 时，按白名单校验模型。
    if model_name:
        app_config = get_app_config()
        resolved = app_config.get_model_config(model_name)
        if resolved is None:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name!r} is not in the configured model allowlist",
            )

    owner_user_id = get_trusted_internal_owner_user_id(request)
    # 无状态 run 端点把 thread_id 放在请求 *体* 中，因此基于路径参数
    # 解析属主的 @require_permission(owner_check=True) 装饰器无法保护
    # 它们。这里在创建任何 run 之前强制执行 thread 属主校验，防止一个
    # 用户在其他用户的 thread 上启动 run（或通过 /wait 读取其 checkpoint
    # 状态）。缺失的行（自动创建的临时 thread）和属主为 NULL 的行
    # （共享 / 认证前的数据）仍可通过 check_access 访问；只有已被其他
    # 用户拥有的 thread 才会以 404 拒绝，与 thread_runs.py 的防枚举行为
    # 一致。内部 channel run 代表 X-DeerFlow-Owner-User-Id 中携带的连接
    # 属主行事，因此按该属主限定范围，而不是跳过校验 —— 泄露的内部
    # 令牌绝不能授予跨用户的 thread 访问权。
    user = getattr(request.state, "user", None)
    if user is not None:
        allowed = await run_ctx.thread_store.check_access(thread_id, str(user.id))
        if not allowed and owner_user_id and getattr(user, "system_role", None) == INTERNAL_SYSTEM_ROLE:
            # channel worker 也可代表受信任头部中指定的连接属主行事
            # （例如为真正的属主认领一个遗留的默认属主 channel thread）。
            allowed = await run_ctx.thread_store.check_access(thread_id, owner_user_id)
        if not allowed:
            raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    owner_context_token = set_current_user(SimpleNamespace(id=owner_user_id)) if owner_user_id else None
    try:
        try:
            record = await run_mgr.create_or_reject(
                thread_id,
                body.assistant_id,
                on_disconnect=disconnect,
                metadata=body.metadata or {},
                kwargs={"input": body.input, "config": body.config},
                multitask_strategy=body.multitask_strategy,
                model_name=model_name,
                user_id=owner_user_id,
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedStrategyError as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc

        # 更新（upsert）thread 元数据，使 thread 出现在 /threads/search
        # 中，即使是从未通过 POST /threads 显式创建的 thread
        # （如无状态 run）。
        try:
            existing = await run_ctx.thread_store.get(thread_id)
            if existing is None and owner_user_id:
                unscoped_existing = await run_ctx.thread_store.get(thread_id, user_id=None)
                if unscoped_existing is not None:
                    if unscoped_existing.get("user_id") != owner_user_id:
                        await run_ctx.thread_store.update_owner(thread_id, owner_user_id, user_id=None)
                    existing = await run_ctx.thread_store.get(thread_id)
            if existing is None:
                await run_ctx.thread_store.create(
                    thread_id,
                    assistant_id=body.assistant_id,
                    metadata=body.metadata,
                )
            else:
                await run_ctx.thread_store.update_status(thread_id, "running")
        except Exception:
            logger.warning("Failed to upsert thread_meta for %s (non-fatal)", sanitize_log_param(thread_id))

        agent_factory = resolve_agent_factory(body.assistant_id)
        graph_input = normalize_input(body.input)
        config = build_run_config(thread_id, body.config, body.metadata, assistant_id=body.assistant_id)

        # 将 DeerFlow 特有的上下文覆盖合并到 ``configurable`` 和 ``context`` 两处。
        # ``context`` 字段是 langgraph-compat 层的自定义扩展，用于携带 agent
        # 配置（model_name、thinking_enabled 等）。只转发与 agent 相关的键；
        # 未知键（如 thread_id）会被忽略。
        merge_run_context_overrides(config, getattr(body, "context", None))
        inject_authenticated_user_context(config, request)

        stream_modes = normalize_stream_modes(body.stream_mode)

        task = asyncio.create_task(
            run_agent(
                bridge,
                run_mgr,
                record,
                ctx=run_ctx,
                agent_factory=agent_factory,
                graph_input=graph_input,
                config=config,
                stream_modes=stream_modes,
                stream_subgraphs=body.stream_subgraphs,
                interrupt_before=body.interrupt_before,
                interrupt_after=body.interrupt_after,
            )
        )
        record.task = task

        # 标题同步由 worker.py 的 finally 块处理：run 完成后从 checkpoint
        # 读取标题并调用 thread_store.update_display_name。

        return record
    finally:
        if owner_context_token is not None:
            reset_current_user(owner_context_token)


async def sse_consumer(
    bridge: StreamBridge,
    record: RunRecord,
    request: Request,
    run_mgr: RunManager,
):
    """从 bridge 产出 SSE 帧的异步生成器。

    ``finally`` 块实现 ``on_disconnect`` 语义：
    - ``cancel``：客户端断开时中止后台任务。
    - ``continue``：任务继续运行，事件被丢弃。
    """
    last_event_id = request.headers.get("Last-Event-ID")
    try:
        async for entry in bridge.subscribe(record.run_id, last_event_id=last_event_id):
            if await request.is_disconnected():
                break

            if entry is HEARTBEAT_SENTINEL:
                yield ": heartbeat\n\n"
                continue

            if entry is END_SENTINEL:
                yield format_sse("end", None, event_id=entry.id or None)
                return

            yield format_sse(entry.event, entry.data, event_id=entry.id or None)

    finally:
        if record.status in (RunStatus.pending, RunStatus.running):
            if record.on_disconnect == DisconnectMode.cancel:
                await run_mgr.cancel(record.run_id)


async def wait_for_run_completion(
    bridge: StreamBridge,
    record: RunRecord,
    request: Request,
    run_mgr: RunManager,
) -> bool:
    """阻塞直到 run 发布 ``END_SENTINEL``，并遵循 on_disconnect 语义。

    非流式的 ``/wait`` 端点过去直接 ``await record.task``，没有任何断连
    处理。当客户端（或中间 HTTP 代理）在 ``pip install`` 之类的长耗时
    工具调用期间超时，处理器会吞掉 ``CancelledError`` 并序列化恰好存在
    的任意 checkpoint —— 把只跑了一半的 run 伪装成正常完成
    （issue #3265）。

    本辅助函数消费与 ``sse_consumer`` 相同的 bridge，使等待路径共享其
    断连语义：每次唤醒都会轮询 ``request.is_disconnected()``；在真实断连
    时，如果 ``record.on_disconnect`` 为 ``cancel`` 则取消后台 run。
    bridge 的心跳哨兵保证即使 agent 长时间不产生事件，每个
    ``heartbeat_interval`` 也至少唤醒一次。

    Returns:
        观察到 ``END_SENTINEL``（run 到达终态）时返回 ``True``；因客户端
        断开而退出循环时返回 ``False``。调用方在收到 ``False`` 时必须跳过
        checkpoint 序列化，避免把不完整的 checkpoint 当作正常响应返回。
    """
    completed = False
    try:
        async for entry in bridge.subscribe(record.run_id):
            # END_SENTINEL 表示 run 到达终态；即使客户端刚刚断开也要尊重
            # 它，使调用方仍能序列化真正的最终 checkpoint。
            if entry is END_SENTINEL:
                completed = True
                return True
            if await request.is_disconnected():
                break
            # 心跳和普通事件：继续等待 END_SENTINEL。
        return completed
    finally:
        if not completed and record.status in (RunStatus.pending, RunStatus.running):
            if record.on_disconnect == DisconnectMode.cancel:
                await run_mgr.cancel(record.run_id)
