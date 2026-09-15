"""Patched ChatDeepSeek that preserves reasoning_content in multi-turn conversations.

This module provides a patched version of ChatDeepSeek that properly handles
reasoning_content when sending messages back to the API. The original implementation
stores reasoning_content in additional_kwargs but doesn't include it when making
subsequent API calls, which causes errors with APIs that require reasoning_content
on all assistant messages when thinking mode is enabled.
"""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_deepseek import ChatDeepSeek
from pydantic import Field

from deerflow.models.assistant_payload_replay import restore_assistant_payloads, restore_reasoning_content


class PatchedChatDeepSeek(ChatDeepSeek):
    """ChatDeepSeek with proper reasoning_content preservation.

    When using thinking/reasoning enabled models, the API expects reasoning_content
    to be present on ALL assistant messages in multi-turn conversations. This patched
    version ensures reasoning_content from additional_kwargs is included in the
    request payload.
    """

    enable_prefix_continuation: bool = Field(
        default=False,
        description=(
            "Append a trailing assistant message with prefix=true to the request "
            "payload so the API continues the assistant turn (DeepSeek 对话前缀续写, Beta). "
            "Requires the /beta base URL, which the model factory switches automatically."
        ),
    )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "DEEPSEEK_API_KEY", "openai_api_key": "DEEPSEEK_API_KEY"}

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get request payload with reasoning_content preserved.

        Overrides the parent method to inject reasoning_content from
        additional_kwargs into assistant messages in the payload.
        """
        # Get the original messages before conversion
        original_messages = self._convert_input(input_).to_messages()

        # Call parent to get the base payload
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        restore_assistant_payloads(
            payload.get("messages", []),
            original_messages,
            restore_reasoning_content,
        )

        if self.enable_prefix_continuation:
            messages = payload.get("messages")
            if isinstance(messages, list):
                last = messages[-1] if messages else None
                # Prefix continuation requires the final message to be an assistant
                # message marked with prefix=true; the model then continues it.
                if not (isinstance(last, dict) and last.get("role") == "assistant"):
                    messages.append({"role": "assistant", "content": "", "prefix": True})

        return payload
