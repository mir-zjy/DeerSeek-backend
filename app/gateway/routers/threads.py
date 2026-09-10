"""Thread 的增删改查、状态与历史接口。

将已有的 thread 本地文件系统清理与基于 checkpointer 的、
兼容 LangGraph Platform 的 thread 管理结合在一起。

状态响应中返回的 channel values 会通过
:func:`deerflow.runtime.serialization.serialize_channel_values` 序列化，
确保 LangChain 消息对象被转换为 JSON 安全的 dict，
符合 ``useStream`` React hook 所期望的 LangGraph Platform 线上格式。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langgraph.checkpoint.base import empty_checkpoint, uuid6
from pydantic import BaseModel, Field, field_validator

from app.gateway.authz import require_permission
from app.gateway.deps import get_checkpointer
from app.gateway.internal_auth import get_trusted_internal_owner_user_id
from app.gateway.utils import sanitize_log_param
from deerflow.config.paths import Paths, get_paths
from deerflow.runtime import serialize_channel_values_for_api
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.utils.time import coerce_iso, now_iso

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/threads", tags=["threads"])


# 由服务端控制的 metadata 键，客户端不允许设置。
# 下方每个入站模型的 Pydantic ``@field_validator("metadata")`` 都会剥离它们，
# 防止恶意客户端通过 API 伪造 owner 身份。属于纵深防御——
# 行级不变量仍是由认证 contextvar 填充的 ``threads_meta.user_id``；
# 此列表用于堵住 metadata 字段回显的漏洞。
_SERVER_RESERVED_METADATA_KEYS: frozenset[str] = frozenset({"owner_id", "user_id"})


def _strip_reserved_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """返回移除服务端控制键后的 ``metadata``。"""
    if not metadata:
        return metadata or {}
    return {k: v for k, v in metadata.items() if k not in _SERVER_RESERVED_METADATA_KEYS}


# ---------------------------------------------------------------------------
# 响应 / 请求模型
# ---------------------------------------------------------------------------


class ThreadDeleteResponse(BaseModel):
    """thread 清理的响应模型。"""

    success: bool
    message: str


class ThreadResponse(BaseModel):
    """单个 thread 的响应模型。"""

    thread_id: str = Field(description="Unique thread identifier")
    status: str = Field(default="idle", description="Thread status: idle, busy, interrupted, error")
    created_at: str = Field(default="", description="ISO timestamp")
    updated_at: str = Field(default="", description="ISO timestamp")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Thread metadata")
    values: dict[str, Any] = Field(default_factory=dict, description="Current state channel values")
    interrupts: dict[str, Any] = Field(default_factory=dict, description="Pending interrupts")


class ThreadCreateRequest(BaseModel):
    """创建 thread 的请求体。"""

    thread_id: str | None = Field(default=None, description="Optional thread ID (auto-generated if omitted)")
    assistant_id: str | None = Field(default=None, description="Associate thread with an assistant")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Initial metadata")

    _strip_reserved = field_validator("metadata")(classmethod(lambda cls, v: _strip_reserved_metadata(v)))


class ThreadSearchRequest(BaseModel):
    """搜索 thread 的请求体。"""

    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata filter (exact match)")
    limit: int = Field(default=100, ge=1, le=1000, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")
    status: str | None = Field(default=None, description="Filter by thread status")

    @field_validator("metadata")
    @classmethod
    def _validate_metadata_filters(cls, v: dict[str, Any]) -> dict[str, Any]:
        """拒绝 SQL 后端无法编译的过滤项。

        保证 SQL 与内存后端行为一致。
        共享校验器见 ``deerflow.persistence.json_compat``。
        """
        if not v:
            return v
        from deerflow.persistence.json_compat import validate_metadata_filter_key, validate_metadata_filter_value

        bad_entries: list[str] = []
        for key, value in v.items():
            if not validate_metadata_filter_key(key):
                bad_entries.append(f"{key!r} (unsafe key)")
            elif not validate_metadata_filter_value(value):
                bad_entries.append(f"{key!r} (unsupported value type {type(value).__name__})")
        if bad_entries:
            raise ValueError(f"Invalid metadata filter entries: {', '.join(bad_entries)}")
        return v


class ThreadStateResponse(BaseModel):
    """thread 状态的响应模型。"""

    values: dict[str, Any] = Field(default_factory=dict, description="Current channel values")
    next: list[str] = Field(default_factory=list, description="Next tasks to execute")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Checkpoint metadata")
    checkpoint: dict[str, Any] = Field(default_factory=dict, description="Checkpoint info")
    checkpoint_id: str | None = Field(default=None, description="Current checkpoint ID")
    parent_checkpoint_id: str | None = Field(default=None, description="Parent checkpoint ID")
    created_at: str | None = Field(default=None, description="Checkpoint timestamp")
    tasks: list[dict[str, Any]] = Field(default_factory=list, description="Interrupted task details")


class ThreadPatchRequest(BaseModel):
    """更新 thread metadata 的请求体。"""

    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata to merge")

    _strip_reserved = field_validator("metadata")(classmethod(lambda cls, v: _strip_reserved_metadata(v)))


class ThreadStateUpdateRequest(BaseModel):
    """更新 thread 状态的请求体（用于 human-in-the-loop 恢复）。"""

    values: dict[str, Any] | None = Field(default=None, description="Channel values to merge")
    checkpoint_id: str | None = Field(default=None, description="Checkpoint to branch from")
    checkpoint: dict[str, Any] | None = Field(default=None, description="Full checkpoint object")
    as_node: str | None = Field(default=None, description="Node identity for the update")


class HistoryEntry(BaseModel):
    """单条 checkpoint 历史记录。"""

    checkpoint_id: str
    parent_checkpoint_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    next: list[str] = Field(default_factory=list)


class ThreadHistoryRequest(BaseModel):
    """checkpoint 历史查询的请求体。"""

    limit: int = Field(default=10, ge=1, le=100, description="Maximum entries")
    before: str | None = Field(default=None, description="Cursor for pagination")


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _delete_thread_data(thread_id: str, paths: Paths | None = None, *, user_id: str | None = None) -> ThreadDeleteResponse:
    """删除某个 thread 的本地持久化文件数据。"""
    path_manager = paths or get_paths()
    try:
        path_manager.delete_thread_dir(thread_id, user_id=user_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError:
        # 非关键问题——磁盘上可能本就没有该 thread 的数据
        logger.debug("No local thread data to delete for %s", sanitize_log_param(thread_id))
        return ThreadDeleteResponse(success=True, message=f"No local data for {thread_id}")
    except Exception as exc:
        logger.exception("Failed to delete thread data for %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to delete local thread data.") from exc

    logger.info("Deleted local thread data for %s", sanitize_log_param(thread_id))
    return ThreadDeleteResponse(success=True, message=f"Deleted local thread data for {thread_id}")


def _derive_thread_status(checkpoint_tuple) -> str:
    """根据 checkpoint metadata 推导 thread 状态。"""
    if checkpoint_tuple is None:
        return "idle"
    pending_writes = getattr(checkpoint_tuple, "pending_writes", None) or []

    # 检查 pending writes 中是否有错误
    for pw in pending_writes:
        if len(pw) >= 2 and pw[1] == "__error__":
            return "error"

    # 检查是否有待执行的 next 任务（表示发生了 interrupt）
    tasks = getattr(checkpoint_tuple, "tasks", None)
    if tasks:
        return "interrupted"

    return "idle"


# ---------------------------------------------------------------------------
# 接口端点
# ---------------------------------------------------------------------------


@router.delete("/{thread_id}", response_model=ThreadDeleteResponse)
@require_permission("threads", "delete", owner_check=True, require_existing=True)
async def delete_thread_data(thread_id: str, request: Request) -> ThreadDeleteResponse:
    """删除某个 thread 的本地持久化文件数据。

    清理 DeerFlow 管理的 thread 目录、移除 checkpoint 数据，
    并从配置的 ThreadMetaStore（sqlite 或 memory）中删除 thread_meta 记录。
    """
    from app.gateway.deps import get_thread_store

    # 清理本地文件系统
    response = _delete_thread_data(thread_id, user_id=get_effective_user_id())

    # 移除 checkpoints（尽力而为）
    checkpointer = getattr(request.app.state, "checkpointer", None)
    if checkpointer is not None:
        try:
            if hasattr(checkpointer, "adelete_thread"):
                await checkpointer.adelete_thread(thread_id)
        except Exception:
            logger.debug("Could not delete checkpoints for thread %s (not critical)", sanitize_log_param(thread_id))

    # 移除 thread_meta 记录（尽力而为）——sqlite 后端必需，
    # 否则被删除的 thread 仍会出现在 /threads/search 中。
    try:
        thread_store = get_thread_store(request)
        await thread_store.delete(thread_id)
    except Exception:
        logger.debug("Could not delete thread_meta for %s (not critical)", sanitize_log_param(thread_id))

    return response


@router.post("", response_model=ThreadResponse)
async def create_thread(body: ThreadCreateRequest, request: Request) -> ThreadResponse:
    """创建新的 thread。

    写入一条 thread_meta 记录（使该 thread 出现在 /threads/search 中）
    以及一个空 checkpoint（使状态接口立即可用）。
    幂等：当 ``thread_id`` 已存在时返回已有记录。
    """
    from app.gateway.deps import get_thread_store

    checkpointer = get_checkpointer(request)
    thread_store = get_thread_store(request)
    thread_id = body.thread_id or str(uuid.uuid4())
    now = now_iso()
    thread_owner_user_id = get_trusted_internal_owner_user_id(request)
    thread_owner_kwargs = {"user_id": thread_owner_user_id} if thread_owner_user_id else {}
    # ``body.metadata`` 已被 ``ThreadCreateRequest._strip_reserved``
    # 剥离服务端保留键——见模型定义。

    # 幂等：已存在时直接返回已有记录
    existing_record = await thread_store.get(thread_id, **thread_owner_kwargs)
    if existing_record is None and thread_owner_user_id:
        unscoped_record = await thread_store.get(thread_id, user_id=None)
        if unscoped_record is not None:
            if unscoped_record.get("user_id") != thread_owner_user_id:
                await thread_store.update_owner(thread_id, thread_owner_user_id, user_id=None)
            existing_record = await thread_store.get(thread_id, **thread_owner_kwargs)
    if existing_record is not None:
        return ThreadResponse(
            thread_id=thread_id,
            status=existing_record.get("status", "idle"),
            created_at=coerce_iso(existing_record.get("created_at", "")),
            updated_at=coerce_iso(existing_record.get("updated_at", "")),
            metadata=existing_record.get("metadata", {}),
        )

    # 写入 thread_meta，使该 thread 立即出现在 /threads/search 中
    try:
        await thread_store.create(
            thread_id,
            assistant_id=getattr(body, "assistant_id", None),
            **thread_owner_kwargs,
            metadata=body.metadata,
        )
    except Exception:
        logger.exception("Failed to write thread_meta for %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to create thread")

    # 写入空 checkpoint，使状态接口立即可用
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        ckpt_metadata = {
            "step": -1,
            "source": "input",
            "writes": None,
            "parents": {},
            **body.metadata,
            "created_at": now,
        }
        await checkpointer.aput(config, empty_checkpoint(), ckpt_metadata, {})
    except Exception:
        logger.exception("Failed to create checkpoint for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to create thread")

    logger.info("Thread created: %s", sanitize_log_param(thread_id))
    return ThreadResponse(
        thread_id=thread_id,
        status="idle",
        created_at=now,
        updated_at=now,
        metadata=body.metadata,
    )


@router.post("/search", response_model=list[ThreadResponse])
async def search_threads(body: ThreadSearchRequest, request: Request) -> list[ThreadResponse]:
    """搜索并列出 thread。

    委托给配置的 ThreadMetaStore 实现
    （sqlite/postgres 走 SQL 后端，内存模式走 Store 后端）。
    """
    from app.gateway.deps import get_thread_store
    from deerflow.persistence.thread_meta import InvalidMetadataFilterError

    repo = get_thread_store(request)
    try:
        rows = await repo.search(
            metadata=body.metadata or None,
            status=body.status,
            limit=body.limit,
            offset=body.offset,
        )
    except InvalidMetadataFilterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [
        ThreadResponse(
            thread_id=r["thread_id"],
            status=r.get("status", "idle"),
            # ``coerce_iso`` 修复 ``MemoryThreadMetaStore`` 历史上用
            # ``time.time()`` 写入的 unix 秒数旧值；
            # SQL 后端返回的本来就是 ISO 字符串，直接透传。
            created_at=coerce_iso(r.get("created_at", "")),
            updated_at=coerce_iso(r.get("updated_at", "")),
            metadata=r.get("metadata", {}),
            values={"title": r["display_name"]} if r.get("display_name") else {},
            interrupts={},
        )
        for r in rows
    ]


@router.patch("/{thread_id}", response_model=ThreadResponse)
@require_permission("threads", "write", owner_check=True, require_existing=True)
async def patch_thread(thread_id: str, body: ThreadPatchRequest, request: Request) -> ThreadResponse:
    """将 metadata 合并进 thread 记录。"""
    from app.gateway.deps import get_thread_store

    thread_store = get_thread_store(request)
    record = await thread_store.get(thread_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    # ``body.metadata`` 已被 ``ThreadPatchRequest._strip_reserved`` 剥离保留键。
    try:
        await thread_store.update_metadata(thread_id, body.metadata)
    except Exception:
        logger.exception("Failed to patch thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to update thread")

    # 重新读取以获取合并后的 metadata 和刷新后的 updated_at
    record = await thread_store.get(thread_id) or record
    return ThreadResponse(
        thread_id=thread_id,
        status=record.get("status", "idle"),
        created_at=coerce_iso(record.get("created_at", "")),
        updated_at=coerce_iso(record.get("updated_at", "")),
        metadata=record.get("metadata", {}),
    )


@router.get("/{thread_id}", response_model=ThreadResponse)
@require_permission("threads", "read", owner_check=True)
async def get_thread(thread_id: str, request: Request) -> ThreadResponse:
    """获取 thread 信息。

    从 ThreadMetaStore 读取 metadata，并从 checkpointer 推导准确的执行状态。
    对于早于 ThreadMetaStore 引入的旧 thread，回退到仅使用
    checkpointer（向后兼容）。
    """
    from app.gateway.deps import get_thread_store

    thread_store = get_thread_store(request)
    checkpointer = get_checkpointer(request)

    record: dict | None = await thread_store.get(thread_id)

    # 从 checkpointer 推导准确的状态
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception:
        logger.exception("Failed to get checkpoint for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to get thread")

    if record is None and checkpoint_tuple is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    # 若 thread 存在于 checkpointer 但不在 thread_meta 中（例如
    # thread_meta 引入之前创建的旧数据），则从 checkpoint metadata
    # 合成一条最小记录。
    if record is None and checkpoint_tuple is not None:
        ckpt_meta = getattr(checkpoint_tuple, "metadata", {}) or {}
        record = {
            "thread_id": thread_id,
            "status": "idle",
            "created_at": coerce_iso(ckpt_meta.get("created_at", "")),
            "updated_at": coerce_iso(ckpt_meta.get("updated_at", ckpt_meta.get("created_at", ""))),
            "metadata": {k: v for k, v in ckpt_meta.items() if k not in ("created_at", "updated_at", "step", "source", "writes", "parents")},
        }

    if record is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    status = _derive_thread_status(checkpoint_tuple) if checkpoint_tuple is not None else record.get("status", "idle")
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {} if checkpoint_tuple is not None else {}
    channel_values = checkpoint.get("channel_values", {})

    return ThreadResponse(
        thread_id=thread_id,
        status=status,
        created_at=coerce_iso(record.get("created_at", "")),
        updated_at=coerce_iso(record.get("updated_at", "")),
        metadata=record.get("metadata", {}),
        values=serialize_channel_values_for_api(channel_values),
    )


# ---------------------------------------------------------------------------
@router.get("/{thread_id}/state", response_model=ThreadStateResponse)
@require_permission("threads", "read", owner_check=True)
async def get_thread_state(thread_id: str, request: Request) -> ThreadStateResponse:
    """获取 thread 的最新状态快照。

    channel values 会被序列化，确保 LangChain 消息对象
    被转换为 JSON 安全的 dict。
    """
    checkpointer = get_checkpointer(request)

    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception:
        logger.exception("Failed to get state for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to get thread state")

    if checkpoint_tuple is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    metadata = getattr(checkpoint_tuple, "metadata", {}) or {}
    checkpoint_id = None
    ckpt_config = getattr(checkpoint_tuple, "config", {})
    if ckpt_config:
        checkpoint_id = ckpt_config.get("configurable", {}).get("checkpoint_id")

    channel_values = checkpoint.get("channel_values", {})

    parent_config = getattr(checkpoint_tuple, "parent_config", None)
    parent_checkpoint_id = None
    if parent_config:
        parent_checkpoint_id = parent_config.get("configurable", {}).get("checkpoint_id")

    tasks_raw = getattr(checkpoint_tuple, "tasks", []) or []
    next_tasks = [t.name for t in tasks_raw if hasattr(t, "name")]
    tasks = [{"id": getattr(t, "id", ""), "name": getattr(t, "name", "")} for t in tasks_raw]

    values = serialize_channel_values_for_api(channel_values)

    return ThreadStateResponse(
        values=values,
        next=next_tasks,
        metadata=metadata,
        checkpoint={"id": checkpoint_id, "ts": coerce_iso(metadata.get("created_at", ""))},
        checkpoint_id=checkpoint_id,
        parent_checkpoint_id=parent_checkpoint_id,
        created_at=coerce_iso(metadata.get("created_at", "")),
        tasks=tasks,
    )


@router.post("/{thread_id}/state", response_model=ThreadStateResponse)
@require_permission("threads", "write", owner_check=True, require_existing=True)
async def update_thread_state(thread_id: str, body: ThreadStateUpdateRequest, request: Request) -> ThreadStateResponse:
    """更新 thread 状态（例如用于 human-in-the-loop 恢复或标题重命名）。

    写入一个新 checkpoint，将 *body.values* 合并进最新的
    channel values，然后通过 ThreadMetaStore 抽象同步更新后的
    ``title`` 字段，使 ``/threads/search`` 在 sqlite 和内存
    后端中都能立即反映该变更。
    """
    from app.gateway.deps import get_thread_store

    checkpointer = get_checkpointer(request)
    thread_store = get_thread_store(request)

    # aput 要求 config 中必须包含 checkpoint_ns——默认为 ""
    # （根图命名空间）。checkpoint_id 可选；省略时获取该 thread
    # 的最新 checkpoint。
    read_config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }
    if body.checkpoint_id:
        read_config["configurable"]["checkpoint_id"] = body.checkpoint_id

    try:
        checkpoint_tuple = await checkpointer.aget_tuple(read_config)
    except Exception:
        logger.exception("Failed to get state for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to get thread state")

    if checkpoint_tuple is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    # 使用可变副本操作，避免意外修改缓存对象。
    checkpoint: dict[str, Any] = dict(getattr(checkpoint_tuple, "checkpoint", {}) or {})
    metadata: dict[str, Any] = dict(getattr(checkpoint_tuple, "metadata", {}) or {})
    channel_values: dict[str, Any] = dict(checkpoint.get("channel_values", {}))

    if body.values:
        channel_values.update(body.values)

    checkpoint["channel_values"] = channel_values
    metadata["updated_at"] = now_iso()

    if body.as_node:
        metadata["source"] = "update"
        metadata["step"] = metadata.get("step", 0) + 1
        metadata["writes"] = {body.as_node: body.values}

    # 分配新的 checkpoint ID，使 aput 执行 INSERT 而不是
    # 原地 REPLACE 已有行。使用 uuid6（按时间有序）而非 uuid4（随机），
    # 使新 ID 在字典序上始终大于前一个——LangGraph 的 checkpointer
    # 通过 max(checkpoint_ids) 的字符串序来确定"最新" checkpoint，
    # 与 uuid6 的时间序一致。
    checkpoint["id"] = str(uuid6())

    # aput 要求 config 中包含 checkpoint_ns——沿用读取时使用的 config
    # （始终包含 checkpoint_ns=""）。新 checkpoint ID 已在上面通过
    # checkpoint["id"] 分配；config 中不要带 checkpoint_id，使写入以
    # 新的 checkpoint 内容为准而不是沿用之前读取的键。
    # 所有受支持的 saver（InMemorySaver、AsyncSqliteSaver、AsyncPostgresSaver）
    # 都会原样保存并回显 checkpoint["id"]——不会自行生成——因此下面的
    # new_config 携带的就是我们在此分配的 uuid6。（由
    # test_update_thread_state_inserts_new_checkpoint_each_call 回归锁定。）
    write_config: dict[str, Any] = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
        }
    }
    try:
        new_config = await checkpointer.aput(write_config, checkpoint, metadata, {})
    except Exception:
        logger.exception("Failed to update state for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to update thread state")

    new_checkpoint_id: str | None = None
    if isinstance(new_config, dict):
        new_checkpoint_id = new_config.get("configurable", {}).get("checkpoint_id")

    # 通过 ThreadMetaStore 抽象同步标题变更，使 /threads/search
    # 在 sqlite 和内存后端中都能立即反映。
    if thread_store and body.values and "title" in body.values:
        new_title = body.values["title"]
        if new_title:  # 跳过空字符串和 None
            try:
                await thread_store.update_display_name(thread_id, new_title)
            except Exception:
                logger.debug("Failed to sync title to thread_meta for %s (non-fatal)", sanitize_log_param(thread_id))

    return ThreadStateResponse(
        values=serialize_channel_values_for_api(channel_values),
        next=[],
        metadata=metadata,
        checkpoint_id=new_checkpoint_id,
        created_at=coerce_iso(metadata.get("created_at", "")),
    )


@router.post("/{thread_id}/history", response_model=list[HistoryEntry])
@require_permission("threads", "read", owner_check=True)
async def get_thread_history(thread_id: str, body: ThreadHistoryRequest, request: Request) -> list[HistoryEntry]:
    """获取 thread 的 checkpoint 历史。

    消息从 checkpointer 的 channel values（权威数据源）读取，
    并通过 :func:`~deerflow.runtime.serialization.serialize_channel_values`
    序列化。只有最新（第一条）checkpoint 携带 ``messages`` 键，
    避免在每条记录中重复。
    """
    checkpointer = get_checkpointer(request)

    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    if body.before:
        config["configurable"]["checkpoint_id"] = body.before

    entries: list[HistoryEntry] = []
    is_latest_checkpoint = True
    try:
        async for checkpoint_tuple in checkpointer.alist(config, limit=body.limit):
            ckpt_config = getattr(checkpoint_tuple, "config", {})
            parent_config = getattr(checkpoint_tuple, "parent_config", None)
            metadata = getattr(checkpoint_tuple, "metadata", {}) or {}
            checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}

            checkpoint_id = ckpt_config.get("configurable", {}).get("checkpoint_id", "")
            parent_id = None
            if parent_config:
                parent_id = parent_config.get("configurable", {}).get("checkpoint_id")

            channel_values = checkpoint.get("channel_values", {})

            # 从 checkpoint 的 channel_values 构建 values
            values: dict[str, Any] = {}
            if title := channel_values.get("title"):
                values["title"] = title
            if thread_data := channel_values.get("thread_data"):
                values["thread_data"] = thread_data

            # 仅在最新 checkpoint 的记录上附带 messages。
            if is_latest_checkpoint:
                messages = channel_values.get("messages")
                if messages:
                    values["messages"] = serialize_channel_values_for_api({"messages": messages}).get("messages", [])
            is_latest_checkpoint = False

            # 推导 next 任务
            tasks_raw = getattr(checkpoint_tuple, "tasks", []) or []
            next_tasks = [t.name for t in tasks_raw if hasattr(t, "name")]

            # 从 metadata 中剥离 LangGraph 内部键
            user_meta = {k: v for k, v in metadata.items() if k not in ("created_at", "updated_at", "step", "source", "writes", "parents")}
            # 保留 step 用于排序参考
            if "step" in metadata:
                user_meta["step"] = metadata["step"]

            entries.append(
                HistoryEntry(
                    checkpoint_id=checkpoint_id,
                    parent_checkpoint_id=parent_id,
                    metadata=user_meta,
                    values=values,
                    created_at=coerce_iso(metadata.get("created_at", "")),
                    next=next_tasks,
                )
            )
    except Exception:
        logger.exception("Failed to get history for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to get thread history")

    return entries
