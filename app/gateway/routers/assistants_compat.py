"""Assistants 兼容接口。

基于 ``langgraph.json`` 图注册表和 ``config.yaml`` 中的 agent 定义，
提供与 LangGraph Platform 兼容的 assistants API。

这是一个最小化的桩实现，用于满足 ``useStream`` React hook 的
初始化要求（``assistants.search()`` 和 ``assistants.get()``）。
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/assistants", tags=["assistants-compat"])


class AssistantResponse(BaseModel):
    assistant_id: str
    graph_id: str
    name: str
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    created_at: str = ""
    updated_at: str = ""
    version: int = 1


class AssistantSearchRequest(BaseModel):
    graph_id: str | None = None
    name: str | None = None
    metadata: dict[str, Any] | None = None
    limit: int = 10
    offset: int = 0


def _get_default_assistant() -> AssistantResponse:
    """返回默认的 lead_agent assistant。"""
    now = datetime.now(UTC).isoformat()
    return AssistantResponse(
        assistant_id="lead_agent",
        graph_id="lead_agent",
        name="lead_agent",
        config={},
        metadata={"created_by": "system"},
        description="DeerFlow lead agent",
        created_at=now,
        updated_at=now,
        version=1,
    )


def _list_assistants() -> list[AssistantResponse]:
    """列出配置中所有可用的 assistant。"""
    assistants = [_get_default_assistant()]

    # 同时包含 config.yaml agents 目录中的自定义 agent
    try:
        from deerflow.config.agents_config import list_custom_agents

        for agent_cfg in list_custom_agents():
            now = datetime.now(UTC).isoformat()
            assistants.append(
                AssistantResponse(
                    assistant_id=agent_cfg.name,
                    graph_id="lead_agent",  # 所有 agent 使用同一个图
                    name=agent_cfg.name,
                    config={},
                    metadata={"created_by": "user"},
                    description=agent_cfg.description or "",
                    created_at=now,
                    updated_at=now,
                    version=1,
                )
            )
    except Exception:
        logger.debug("Could not load custom agents for assistants list")

    return assistants


@router.post("/search", response_model=list[AssistantResponse])
async def search_assistants(body: AssistantSearchRequest | None = None) -> list[AssistantResponse]:
    """搜索 assistant。

    返回所有已注册的 assistant（lead_agent 加上配置中的自定义 agent）。
    """
    assistants = _list_assistants()

    if body and body.graph_id:
        assistants = [a for a in assistants if a.graph_id == body.graph_id]
    if body and body.name:
        assistants = [a for a in assistants if body.name.lower() in a.name.lower()]

    offset = body.offset if body else 0
    limit = body.limit if body else 10
    return assistants[offset : offset + limit]


@router.get("/{assistant_id}", response_model=AssistantResponse)
async def get_assistant_compat(assistant_id: str) -> AssistantResponse:
    """按 ID 获取 assistant。"""
    for a in _list_assistants():
        if a.assistant_id == assistant_id:
            return a
    raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")


@router.get("/{assistant_id}/graph")
async def get_assistant_graph(assistant_id: str) -> dict:
    """获取 assistant 的图结构。

    返回最小化的图描述。Gateway 不支持完整的图内省，
    此桩实现仅用于满足 SDK 校验。
    """
    found = any(a.assistant_id == assistant_id for a in _list_assistants())
    if not found:
        raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")

    return {
        "graph_id": "lead_agent",
        "nodes": [],
        "edges": [],
    }


@router.get("/{assistant_id}/schemas")
async def get_assistant_schemas(assistant_id: str) -> dict:
    """获取 assistant 输入/输出/状态的 JSON schema。

    返回空 schema —— Gateway 不支持完整内省。
    """
    found = any(a.assistant_id == assistant_id for a in _list_assistants())
    if not found:
        raise HTTPException(status_code=404, detail=f"Assistant {assistant_id} not found")

    return {
        "graph_id": "lead_agent",
        "input_schema": {},
        "output_schema": {},
        "state_schema": {},
        "config_schema": {},
    }
