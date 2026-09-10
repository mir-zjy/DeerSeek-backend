import json
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.gateway.deps import require_admin_user
from deerflow.config.extensions_config import ExtensionsConfig, get_extensions_config, reload_extensions_config
from deerflow.mcp.cache import reset_mcp_tools_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["mcp"])

_ADMIN_REQUIRED_DETAIL = "Admin privileges required to manage MCP configuration."


_MCP_STDIO_COMMAND_ALLOWLIST_ENV = "DEER_FLOW_MCP_STDIO_COMMAND_ALLOWLIST"
_DEFAULT_MCP_STDIO_COMMAND_ALLOWLIST = frozenset({"npx", "uvx"})
_SHELL_METACHARS = frozenset(";|&`$<>\n\r")


class McpOAuthConfigResponse(BaseModel):
    """MCP 服务器的 OAuth 配置。"""

    enabled: bool = Field(default=True, description="Whether OAuth token injection is enabled")
    token_url: str = Field(default="", description="OAuth token endpoint URL")
    grant_type: Literal["client_credentials", "refresh_token"] = Field(default="client_credentials", description="OAuth grant type")
    client_id: str | None = Field(default=None, description="OAuth client ID")
    client_secret: str | None = Field(default=None, description="OAuth client secret")
    refresh_token: str | None = Field(default=None, description="OAuth refresh token")
    scope: str | None = Field(default=None, description="OAuth scope")
    audience: str | None = Field(default=None, description="OAuth audience")
    token_field: str = Field(default="access_token", description="Token response field containing access token")
    token_type_field: str = Field(default="token_type", description="Token response field containing token type")
    expires_in_field: str = Field(default="expires_in", description="Token response field containing expires-in seconds")
    default_token_type: str = Field(default="Bearer", description="Default token type when response omits token_type")
    refresh_skew_seconds: int = Field(default=60, description="Refresh this many seconds before expiry")
    extra_token_params: dict[str, str] = Field(default_factory=dict, description="Additional form params sent to token endpoint")


class McpServerConfigResponse(BaseModel):
    """MCP 服务器配置的响应模型。"""

    enabled: bool = Field(default=True, description="Whether this MCP server is enabled")
    type: str = Field(default="stdio", description="Transport type: 'stdio', 'sse', or 'http'")
    command: str | None = Field(default=None, description="Command to execute to start the MCP server (for stdio type)")
    args: list[str] = Field(default_factory=list, description="Arguments to pass to the command (for stdio type)")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for the MCP server")
    url: str | None = Field(default=None, description="URL of the MCP server (for sse or http type)")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP headers to send (for sse or http type)")
    oauth: McpOAuthConfigResponse | None = Field(default=None, description="OAuth configuration for MCP HTTP/SSE servers")
    description: str = Field(default="", description="Human-readable description of what this MCP server provides")


class McpConfigResponse(BaseModel):
    """MCP 配置的响应模型。"""

    mcp_servers: dict[str, McpServerConfigResponse] = Field(
        default_factory=dict,
        description="Map of MCP server name to configuration",
    )


class McpConfigUpdateRequest(BaseModel):
    """更新 MCP 配置的请求模型。"""

    mcp_servers: dict[str, McpServerConfigResponse] = Field(
        ...,
        description="Map of MCP server name to configuration",
    )


class McpCacheResetResponse(BaseModel):
    """重置 MCP 工具缓存的响应模型。"""

    success: bool = Field(description="Whether the MCP tools cache was reset")
    message: str = Field(description="Human-readable reset status")


_MASKED_VALUE = "***"


def _allowed_stdio_commands() -> set[str]:
    """返回允许通过 API 管理的 stdio MCP 服务器可执行文件名。"""
    raw = os.environ.get(_MCP_STDIO_COMMAND_ALLOWLIST_ENV)
    base = set(_DEFAULT_MCP_STDIO_COMMAND_ALLOWLIST)
    if raw is None:
        return base
    extra = {item.strip() for item in raw.split(",") if item.strip()}
    return base | extra


def _stdio_command_name(command: str | None, *, server_name: str) -> str:
    """规范化并校验 API 边界传入的 stdio command 字段。"""
    if command is None or not command.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MCP server '{server_name}' with stdio transport requires a command.",
        )

    stripped = command.strip()
    has_path_separator = "/" in stripped or "\\" in stripped
    if stripped != command or has_path_separator or any(ch.isspace() for ch in stripped) or any(ch in stripped for ch in _SHELL_METACHARS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(f"MCP server '{server_name}' command must be a single executable name; put parameters in args instead."),
        )

    return stripped


def _validate_mcp_update_request(request: McpConfigUpdateRequest) -> None:
    """在持久化之前校验通过 API 提交的 MCP 配置。

    本地配置文件仍可表达任意高级配置，但 HTTP API 是不可信边界。
    在此限制 stdio 命令可缩小已认证浏览器会话被攻破后的影响范围。
    """
    allowed_commands = _allowed_stdio_commands()
    for name, server in request.mcp_servers.items():
        transport_type = (server.type or "stdio").lower()
        if transport_type != "stdio":
            continue

        command_name = _stdio_command_name(server.command, server_name=name)
        if command_name not in allowed_commands:
            allowed = ", ".join(sorted(allowed_commands)) or "<none>"
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(f"MCP server '{name}' uses disallowed stdio command '{command_name}'. Allowed commands: {allowed}. Configure {_MCP_STDIO_COMMAND_ALLOWLIST_ENV} to extend this list."),
            )


def _mask_server_config(server: McpServerConfigResponse) -> McpServerConfigResponse:
    """返回敏感字段已脱敏的服务器配置副本。

    对 env 值、header 值进行掩码，并移除 OAuth 密钥，
    避免它们通过 GET API 端点泄露。
    """
    masked_env = {k: _MASKED_VALUE for k in server.env}
    masked_headers = {k: _MASKED_VALUE for k in server.headers}
    masked_oauth = None
    if server.oauth is not None:
        masked_oauth = server.oauth.model_copy(
            update={
                "client_secret": None,
                "refresh_token": None,
            }
        )
    return server.model_copy(
        update={
            "env": masked_env,
            "headers": masked_headers,
            "oauth": masked_oauth,
        }
    )


def _merge_preserving_secrets(
    incoming: McpServerConfigResponse,
    existing: McpServerConfigResponse,
) -> McpServerConfigResponse:
    """合并传入配置与现有配置，保留被 GET 脱敏的密钥。

    前端切换 ``enabled`` 时会整体往返配置：
    GET（脱敏）→ 修改 enabled → PUT（带回脱敏值）。
    本函数确保掩码值（``***``）被替换为当前磁盘配置中的真实密钥。

    只有已存在于 *existing* 中的键才接受 ``***``。
    新键必须提供真实值。

    对 OAuth 密钥，``None`` 表示“保留已有存储值”，
    以便脱敏的 GET 响应可以安全往返。要显式清除已存储的密钥，
    客户端可发送空字符串，持久化前会被转换为 ``None``。
    """
    merged_env = {}
    for k, v in incoming.env.items():
        if v == _MASKED_VALUE:
            if k in existing.env:
                merged_env[k] = existing.env[k]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot set env key '{k}' to masked value '***'; provide a real value.",
                )
        else:
            merged_env[k] = v

    merged_headers = {}
    for k, v in incoming.headers.items():
        if v == _MASKED_VALUE:
            if k in existing.headers:
                merged_headers[k] = existing.headers[k]
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot set header '{k}' to masked value '***'; provide a real value.",
                )
        else:
            merged_headers[k] = v

    merged_oauth = incoming.oauth
    if incoming.oauth is not None and existing.oauth is not None:
        # None = 保留（脱敏往返），"" = 显式清除，其他 = 新值
        merged_client_secret = existing.oauth.client_secret if incoming.oauth.client_secret is None else (None if incoming.oauth.client_secret == "" else incoming.oauth.client_secret)
        merged_refresh_token = existing.oauth.refresh_token if incoming.oauth.refresh_token is None else (None if incoming.oauth.refresh_token == "" else incoming.oauth.refresh_token)
        merged_oauth = incoming.oauth.model_copy(
            update={
                "client_secret": merged_client_secret,
                "refresh_token": merged_refresh_token,
            }
        )
    return incoming.model_copy(
        update={
            "env": merged_env,
            "headers": merged_headers,
            "oauth": merged_oauth,
        }
    )


@router.get(
    "/mcp/config",
    response_model=McpConfigResponse,
    summary="Get MCP Configuration",
    description="Retrieve the current Model Context Protocol (MCP) server configurations.",
)
async def get_mcp_configuration(request: Request) -> McpConfigResponse:
    """获取当前 MCP 配置。

    Returns:
        包含所有服务器的当前 MCP 配置。

    Example:
        ```json
        {
            "mcp_servers": {
                "github": {
                    "enabled": true,
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "***"},
                    "description": "GitHub MCP server for repository operations"
                }
            }
        }
        ```
    """
    await require_admin_user(request, detail=_ADMIN_REQUIRED_DETAIL)

    config = get_extensions_config()

    servers = {name: _mask_server_config(McpServerConfigResponse(**server.model_dump())) for name, server in config.mcp_servers.items()}
    return McpConfigResponse(mcp_servers=servers)


@router.post(
    "/mcp/cache/reset",
    response_model=McpCacheResetResponse,
    summary="Reset MCP Tools Cache",
    description=("Reset cached MCP tools and pooled sessions process-wide so tools are reloaded on next use. This affects all threads and users in the current Gateway process."),
)
async def reset_mcp_tools_cache_endpoint(request: Request) -> McpCacheResetResponse:
    """进程级重置缓存的 MCP 工具和持久会话。

    下一次 agent 运行或工具查询将从配置的 MCP 服务器重新加载工具。
    这会影响当前 Gateway 进程中的所有线程和用户，
    且不再依赖 extensions_config.json 的 mtime 变化。
    """
    await require_admin_user(request, detail=_ADMIN_REQUIRED_DETAIL)
    reset_mcp_tools_cache()
    return McpCacheResetResponse(
        success=True,
        message="MCP tools cache reset. Tools will reload on next use.",
    )


@router.put(
    "/mcp/config",
    response_model=McpConfigResponse,
    summary="Update MCP Configuration",
    description="Update Model Context Protocol (MCP) server configurations and save to file.",
)
async def update_mcp_configuration(request: Request, body: McpConfigUpdateRequest) -> McpConfigResponse:
    """更新 MCP 配置。

    这将：
    1. 将新配置保存到 mcp_config.json 文件
    2. 重新加载配置缓存
    3. 重置 MCP 工具缓存以触发重新初始化

    Args:
        request: 要保存的新 MCP 配置。

    Returns:
        更新后的 MCP 配置。

    Raises:
        HTTPException: 配置文件无法写入时返回 500。

    Example Request:
        ```json
        {
            "mcp_servers": {
                "github": {
                    "enabled": true,
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": {"GITHUB_TOKEN": "$GITHUB_TOKEN"},
                    "description": "GitHub MCP server for repository operations"
                }
            }
        }
        ```
    """
    try:
        await require_admin_user(request, detail=_ADMIN_REQUIRED_DETAIL)
        _validate_mcp_update_request(body)

        # 获取当前配置路径（或确定保存位置）
        config_path = ExtensionsConfig.resolve_config_path()

        # 如果配置文件不存在，则在工作目录（项目根目录 backend/）创建一个
        if config_path is None:
            config_path = Path.cwd() / "extensions_config.json"
            logger.info(f"No existing extensions config found. Creating new config at: {config_path}")

        # 加载当前配置以保留 skills
        current_config = get_extensions_config()

        # 从磁盘加载原始（未解析变量）的 JSON 作为合并来源。
        # 这可保留 env 值中的 $VAR 占位符以及 mcpInterceptors 等
        # 顶层键，否则它们会丢失。
        raw_servers: dict[str, dict] = {}
        raw_other_keys: dict = {}
        if config_path is not None and config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw_data = json.load(f)
            raw_servers = raw_data.get("mcpServers", {})
            # 保留 mcpServers/skills 之外的所有顶层键
            for key, value in raw_data.items():
                if key not in ("mcpServers", "skills"):
                    raw_other_keys[key] = value

        # 将传入的服务器配置与磁盘上的原始密钥合并
        merged_servers: dict[str, McpServerConfigResponse] = {}
        for name, incoming in body.mcp_servers.items():
            raw_server = raw_servers.get(name)
            if raw_server is not None:
                merged_servers[name] = _merge_preserving_secrets(
                    incoming,
                    McpServerConfigResponse(**raw_server),
                )
            else:
                merged_servers[name] = incoming

        # 构建配置数据，保留原文件中的所有顶层键
        config_data = dict(raw_other_keys)
        config_data["mcpServers"] = {name: server.model_dump() for name, server in merged_servers.items()}
        config_data["skills"] = {name: {"enabled": skill.enabled} for name, skill in current_config.skills.items()}

        # 将配置写入文件
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"MCP configuration updated and saved to: {config_path}")

        # 重新加载 Gateway 配置并更新全局缓存。agent 运行时在 Gateway 中，
        # 这样在 extensions_config.json 变更后，API 读取与工具执行保持一致。
        reloaded_config = reload_extensions_config()
        reset_mcp_tools_cache()
        servers = {name: _mask_server_config(McpServerConfigResponse(**server.model_dump())) for name, server in reloaded_config.mcp_servers.items()}
        return McpConfigResponse(mcp_servers=servers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update MCP configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update MCP configuration: {str(e)}")
