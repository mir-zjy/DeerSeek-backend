"""自定义 agent 的 CRUD API。"""

import asyncio
import logging
import re
import shutil

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deerflow.config.agents_api_config import get_agents_api_config
from deerflow.config.agents_config import AgentConfig, list_custom_agents, load_agent_config, load_agent_soul
from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["agents"])

AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


class AgentResponse(BaseModel):
    """自定义 agent 的响应模型。"""

    name: str = Field(..., description="Agent name (hyphen-case)")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    skills: list[str] | None = Field(default=None, description="Optional skill whitelist (None=all, []=none)")
    soul: str | None = Field(default=None, description="SOUL.md content")


class AgentsListResponse(BaseModel):
    """列出所有自定义 agent 的响应模型。"""

    agents: list[AgentResponse]


class AgentCreateRequest(BaseModel):
    """创建自定义 agent 的请求体。"""

    name: str = Field(..., description="Agent name (must match ^[A-Za-z0-9-]+$, stored as lowercase)")
    description: str = Field(default="", description="Agent description")
    model: str | None = Field(default=None, description="Optional model override")
    tool_groups: list[str] | None = Field(default=None, description="Optional tool group whitelist")
    skills: list[str] | None = Field(default=None, description="Optional skill whitelist (None=all enabled, []=none)")
    soul: str = Field(default="", description="SOUL.md content — agent personality and behavioral guardrails")


class AgentUpdateRequest(BaseModel):
    """更新自定义 agent 的请求体。"""

    description: str | None = Field(default=None, description="Updated description")
    model: str | None = Field(default=None, description="Updated model override")
    tool_groups: list[str] | None = Field(default=None, description="Updated tool group whitelist")
    skills: list[str] | None = Field(default=None, description="Updated skill whitelist (None=all, []=none)")
    soul: str | None = Field(default=None, description="Updated SOUL.md content")


def _validate_agent_name(name: str) -> None:
    """按允许的格式校验 agent 名称。

    Args:
        name: 待校验的 agent 名称。

    Raises:
        HTTPException: 名称不合法时返回 422。
    """
    if not AGENT_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid agent name '{name}'. Must match ^[A-Za-z0-9-]+$ (letters, digits, and hyphens only).",
        )


def _normalize_agent_name(name: str) -> str:
    """将 agent 名称规范化为小写，以便存入文件系统。"""
    return name.lower()


def _require_agents_api_enabled() -> None:
    """除非显式启用了自定义 agent 管理 API，否则拒绝访问。"""
    if not get_agents_api_config().enabled:
        raise HTTPException(
            status_code=403,
            detail=("Custom-agent management API is disabled. Set agents_api.enabled=true to expose agent and user-profile routes over HTTP."),
        )


def _agent_config_to_response(agent_cfg: AgentConfig, include_soul: bool = False, *, user_id: str | None = None) -> AgentResponse:
    """将 AgentConfig 转换为 AgentResponse。"""
    soul: str | None = None
    if include_soul:
        soul = load_agent_soul(agent_cfg.name, user_id=user_id) or ""

    return AgentResponse(
        name=agent_cfg.name,
        description=agent_cfg.description,
        model=agent_cfg.model,
        tool_groups=agent_cfg.tool_groups,
        skills=agent_cfg.skills,
        soul=soul,
    )


@router.get(
    "/agents",
    response_model=AgentsListResponse,
    summary="List Custom Agents",
    description="List all custom agents available in the agents directory, including their soul content.",
)
async def list_agents() -> AgentsListResponse:
    """列出所有自定义 agent。

    Returns:
        包含所有自定义 agent 元数据及 soul 内容的列表。
    """
    _require_agents_api_enabled()

    user_id = get_effective_user_id()
    try:
        agents = list_custom_agents(user_id=user_id)
        return AgentsListResponse(agents=[_agent_config_to_response(a, include_soul=True, user_id=user_id) for a in agents])
    except Exception as e:
        logger.error(f"Failed to list agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list agents: {str(e)}")


@router.get(
    "/agents/check",
    summary="Check Agent Name",
    description="Validate an agent name and check if it is available (case-insensitive).",
)
async def check_agent_name(name: str) -> dict:
    """检查 agent 名称是否合法且未被占用。

    Args:
        name: 待检查的 agent 名称。

    Returns:
        ``{"available": true/false, "name": "<normalized>"}``

    Raises:
        HTTPException: 名称不合法时返回 422。
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    normalized = _normalize_agent_name(name)
    user_id = get_effective_user_id()
    paths = get_paths()
    # 只要按用户路径或旧版共享路径中存在同名 agent，就视为已占用。
    # 选择与尚未迁移的旧版 agent 冲突的名称，迁移后会遮蔽旧条目。
    available = not paths.user_agent_dir(user_id, normalized).exists() and not paths.agent_dir(normalized).exists()
    return {"available": available, "name": normalized}


@router.get(
    "/agents/{name}",
    response_model=AgentResponse,
    summary="Get Custom Agent",
    description="Retrieve details and SOUL.md content for a specific custom agent.",
)
async def get_agent(name: str) -> AgentResponse:
    """按名称获取指定的自定义 agent。

    Args:
        name: agent 名称。

    Returns:
        agent 详情，包括 SOUL.md 内容。

    Raises:
        HTTPException: agent 不存在时返回 404。
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    name = _normalize_agent_name(name)
    user_id = get_effective_user_id()

    try:
        agent_cfg = load_agent_config(name, user_id=user_id)
        return _agent_config_to_response(agent_cfg, include_soul=True, user_id=user_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")
    except Exception as e:
        logger.error(f"Failed to get agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get agent: {str(e)}")


@router.post(
    "/agents",
    response_model=AgentResponse,
    status_code=201,
    summary="Create Custom Agent",
    description="Create a new custom agent with its config and SOUL.md.",
)
async def create_agent_endpoint(request: AgentCreateRequest) -> AgentResponse:
    """创建新的自定义 agent。

    Args:
        request: agent 创建请求。

    Returns:
        创建后的 agent 详情。

    Raises:
        HTTPException: agent 已存在时返回 409，名称不合法时返回 422。
    """
    _require_agents_api_enabled()
    _validate_agent_name(request.name)
    normalized_name = _normalize_agent_name(request.name)
    user_id = get_effective_user_id()
    paths = get_paths()

    def _create_agent() -> AgentResponse | None:
        # 工作线程：基目录解析、存在性检查、目录/文件创建、回读以及
        # 失败清理都是阻塞式文件系统 IO，必须放在事件循环之外。
        agent_dir = paths.user_agent_dir(user_id, normalized_name)
        legacy_dir = paths.agent_dir(normalized_name)

        if legacy_dir.exists():
            return None  # 向调用方返回 409 的信号

        try:
            try:
                agent_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                return None  # 向调用方返回 409 的信号
            # 写入 config.yaml
            config_data: dict = {"name": normalized_name}
            if request.description:
                config_data["description"] = request.description
            if request.model is not None:
                config_data["model"] = request.model
            if request.tool_groups is not None:
                config_data["tool_groups"] = request.tool_groups
            if request.skills is not None:
                config_data["skills"] = request.skills

            config_file = agent_dir / "config.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

            # 写入 SOUL.md
            soul_file = agent_dir / "SOUL.md"
            soul_file.write_text(request.soul, encoding="utf-8")

            logger.info(f"Created agent '{normalized_name}' at {agent_dir}")

            agent_cfg = load_agent_config(normalized_name, user_id=user_id)
            return _agent_config_to_response(agent_cfg, include_soul=True, user_id=user_id)
        except Exception:
            # 抛出错误前先清理失败产生的部分状态。
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            raise

    try:
        response = await asyncio.to_thread(_create_agent)
    except Exception as e:
        logger.error(f"Failed to create agent '{request.name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")

    if response is None:
        raise HTTPException(status_code=409, detail=f"Agent '{normalized_name}' already exists")

    return response


@router.put(
    "/agents/{name}",
    response_model=AgentResponse,
    summary="Update Custom Agent",
    description="Update an existing custom agent's config and/or SOUL.md.",
)
async def update_agent(name: str, request: AgentUpdateRequest) -> AgentResponse:
    """更新已有的自定义 agent。

    Args:
        name: agent 名称。
        request: 更新请求（所有字段均可选）。

    Returns:
        更新后的 agent 详情。

    Raises:
        HTTPException: agent 不存在时返回 404。
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    name = _normalize_agent_name(name)
    user_id = get_effective_user_id()

    try:
        agent_cfg = load_agent_config(name, user_id=user_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    paths = get_paths()
    agent_dir = paths.user_agent_dir(user_id, name)
    if not agent_dir.exists() and paths.agent_dir(name).exists():
        raise HTTPException(
            status_code=409,
            detail=(f"Agent '{name}' only exists in the legacy shared layout and is not scoped to a user. Run scripts/migrate_user_isolation.py to move legacy agents into the per-user layout before updating."),
        )

    try:
        # 如果任何配置字段有变化，则更新配置
        # 使用 model_fields_set 区分“字段未提供”和“显式设为 null”。
        # 这对 skills 很关键：None 表示“全部继承”（而非“不修改”）。
        fields_set = request.model_fields_set
        config_changed = bool(fields_set & {"description", "model", "tool_groups", "skills"})

        if config_changed:
            updated: dict = {
                "name": agent_cfg.name,
                "description": request.description if "description" in fields_set else agent_cfg.description,
            }
            new_model = request.model if "model" in fields_set else agent_cfg.model
            if new_model is not None:
                updated["model"] = new_model

            new_tool_groups = request.tool_groups if "tool_groups" in fields_set else agent_cfg.tool_groups
            if new_tool_groups is not None:
                updated["tool_groups"] = new_tool_groups

            # skills: None = 全部继承，[] = 不启用任何 skill，["a","b"] = 白名单
            if "skills" in fields_set:
                new_skills = request.skills
            else:
                new_skills = agent_cfg.skills
            if new_skills is not None:
                updated["skills"] = new_skills

            config_file = agent_dir / "config.yaml"
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(updated, f, default_flow_style=False, allow_unicode=True)

        # 如果提供了 soul，则更新 SOUL.md
        if request.soul is not None:
            soul_path = agent_dir / "SOUL.md"
            soul_path.write_text(request.soul, encoding="utf-8")

        logger.info(f"Updated agent '{name}'")

        refreshed_cfg = load_agent_config(name, user_id=user_id)
        return _agent_config_to_response(refreshed_cfg, include_soul=True, user_id=user_id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update agent: {str(e)}")


class UserProfileResponse(BaseModel):
    """全局用户档案（USER.md）的响应模型。"""

    content: str | None = Field(default=None, description="USER.md content, or null if not yet created")


class UserProfileUpdateRequest(BaseModel):
    """设置全局用户档案的请求体。"""

    content: str = Field(default="", description="USER.md content — describes the user's background and preferences")


@router.get(
    "/user-profile",
    response_model=UserProfileResponse,
    summary="Get User Profile",
    description="Read the global USER.md file that is injected into all custom agents.",
)
async def get_user_profile() -> UserProfileResponse:
    """返回当前 USER.md 的内容。

    Returns:
        USER.md 尚不存在时，返回 content=None 的 UserProfileResponse。
    """
    _require_agents_api_enabled()

    try:
        user_md_path = get_paths().user_md_file
        if not user_md_path.exists():
            return UserProfileResponse(content=None)
        raw = user_md_path.read_text(encoding="utf-8").strip()
        return UserProfileResponse(content=raw or None)
    except Exception as e:
        logger.error(f"Failed to read user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read user profile: {str(e)}")


@router.put(
    "/user-profile",
    response_model=UserProfileResponse,
    summary="Update User Profile",
    description="Write the global USER.md file that is injected into all custom agents.",
)
async def update_user_profile(request: UserProfileUpdateRequest) -> UserProfileResponse:
    """创建或覆盖全局 USER.md。

    Args:
        request: 包含新 USER.md 内容的更新请求。

    Returns:
        包含已保存内容的 UserProfileResponse。
    """
    _require_agents_api_enabled()

    try:
        paths = get_paths()
        paths.base_dir.mkdir(parents=True, exist_ok=True)
        paths.user_md_file.write_text(request.content, encoding="utf-8")
        logger.info(f"Updated USER.md at {paths.user_md_file}")
        return UserProfileResponse(content=request.content or None)
    except Exception as e:
        logger.error(f"Failed to update user profile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update user profile: {str(e)}")


@router.delete(
    "/agents/{name}",
    status_code=204,
    summary="Delete Custom Agent",
    description="Delete a custom agent and all its files (config, SOUL.md, memory).",
)
async def delete_agent(name: str) -> None:
    """删除一个自定义 agent。

    Args:
        name: agent 名称。

    Raises:
        HTTPException: 无按用户的副本时返回 404；仅存在旧版共享副本时
            返回 409（提示运行迁移脚本）。
    """
    _require_agents_api_enabled()
    _validate_agent_name(name)
    name = _normalize_agent_name(name)
    user_id = get_effective_user_id()
    paths = get_paths()

    def _remove_agent_dir() -> tuple[str, str]:
        # 在工作线程中运行：解析基目录、探测目录（`exists`）、
        # 删除目录（`rmtree`）都是阻塞式文件系统 IO，必须放在事件循环之外。
        agent_dir = paths.user_agent_dir(user_id, name)
        if not agent_dir.exists():
            outcome = "legacy" if paths.agent_dir(name).exists() else "missing"
            return outcome, str(agent_dir)
        shutil.rmtree(agent_dir)
        return "deleted", str(agent_dir)

    try:
        outcome, agent_dir = await asyncio.to_thread(_remove_agent_dir)
    except Exception as e:
        logger.error(f"Failed to delete agent '{name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete agent: {str(e)}")

    if outcome == "legacy":
        raise HTTPException(
            status_code=409,
            detail=(f"Agent '{name}' only exists in the legacy shared layout and is not scoped to a user. Run scripts/migrate_user_isolation.py to move legacy agents into the per-user layout before deleting."),
        )
    if outcome == "missing":
        raise HTTPException(status_code=404, detail=f"Agent '{name}' not found")

    logger.info(f"Deleted agent '{name}' from {agent_dir}")
