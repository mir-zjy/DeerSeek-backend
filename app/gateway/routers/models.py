from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.gateway.deps import get_config
from deerflow.config.app_config import AppConfig

router = APIRouter(prefix="/api", tags=["models"])


class ModelResponse(BaseModel):
    """模型信息的响应模型。"""

    name: str = Field(..., description="Unique identifier for the model")
    model: str = Field(..., description="Actual provider model identifier")
    display_name: str | None = Field(None, description="Human-readable name")
    description: str | None = Field(None, description="Model description")
    supports_thinking: bool = Field(default=False, description="Whether model supports thinking mode")
    supports_reasoning_effort: bool = Field(default=False, description="Whether model supports reasoning effort")
    supports_vision: bool = Field(default=False, description="Whether model supports vision/image inputs")
    supports_json_output: bool = Field(default=False, description="Whether model supports JSON Output")
    supports_prefix_continuation: bool = Field(default=False, description="Whether model supports chat prefix continuation (Beta)")
    supports_fim: bool = Field(default=False, description="Whether model supports FIM completion (Beta, non-thinking only)")
    context_window: int | None = Field(None, description="Context window size in tokens")
    max_output_length: int | None = Field(None, description="Maximum output length in tokens")


def _to_model_response(model) -> ModelResponse:
    """将 ModelConfig 序列化为前端展示用的 ModelResponse（不含敏感字段）。"""
    return ModelResponse(
        name=model.name,
        model=model.model,
        display_name=model.display_name,
        description=model.description,
        supports_thinking=model.supports_thinking,
        supports_reasoning_effort=model.supports_reasoning_effort,
        supports_vision=model.supports_vision,
        supports_json_output=model.supports_json_output,
        supports_prefix_continuation=model.supports_prefix_continuation,
        supports_fim=model.supports_fim,
        context_window=model.context_window,
        max_output_length=model.max_output_length,
    )


class TokenUsageResponse(BaseModel):
    """Token 用量展示配置。"""

    enabled: bool = Field(default=False, description="Whether token usage display is enabled")


class ModelsListResponse(BaseModel):
    """列出所有模型的响应模型。"""

    models: list[ModelResponse]
    token_usage: TokenUsageResponse


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List All Models",
    description="Retrieve a list of all available AI models configured in the system.",
)
async def list_models(config: AppConfig = Depends(get_config)) -> ModelsListResponse:
    """列出配置中所有可用的模型。

    返回适合前端展示的模型信息，
    不包含 API key 和内部配置等敏感字段。

    Returns:
        所有已配置模型及其元数据和 token 用量展示设置的列表。

    Example Response:
        ```json
        {
            "models": [
                {
                    "name": "gpt-4",
                    "model": "gpt-4",
                    "display_name": "GPT-4",
                    "description": "OpenAI GPT-4 model",
                    "supports_thinking": false,
                    "supports_reasoning_effort": false
                },
                {
                    "name": "claude-3-opus",
                    "model": "claude-3-opus",
                    "display_name": "Claude 3 Opus",
                    "description": "Anthropic Claude 3 Opus model",
                    "supports_thinking": true,
                    "supports_reasoning_effort": false
                }
            ],
            "token_usage": {
                "enabled": true
            }
        }
        ```
    """
    models = [_to_model_response(model) for model in config.models]
    return ModelsListResponse(
        models=models,
        token_usage=TokenUsageResponse(enabled=config.token_usage.enabled),
    )


@router.get(
    "/models/{model_name}",
    response_model=ModelResponse,
    summary="Get Model Details",
    description="Retrieve detailed information about a specific AI model by its name.",
)
async def get_model(model_name: str, config: AppConfig = Depends(get_config)) -> ModelResponse:
    """按名称获取指定模型。

    Args:
        model_name: 要获取的模型的唯一名称。

    Returns:
        找到时返回模型信息。

    Raises:
        HTTPException: 模型未找到时返回 404。

    Example Response:
        ```json
        {
            "name": "gpt-4",
            "display_name": "GPT-4",
            "description": "OpenAI GPT-4 model",
            "supports_thinking": false
        }
        ```
    """
    model = config.get_model_config(model_name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    return _to_model_response(model)
