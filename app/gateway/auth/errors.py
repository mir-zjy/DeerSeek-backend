"""auth 模块的类型化错误定义。

AuthErrorCode: 所有认证失败情况的完整枚举。
TokenError: JWT 解码失败原因的完整枚举。
AuthErrorResponse: HTTP 响应使用的结构化错误负载。
"""

from enum import StrEnum

from pydantic import BaseModel


class AuthErrorCode(StrEnum):
    """认证错误情况的完整列表。"""

    INVALID_CREDENTIALS = "invalid_credentials"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"
    USER_NOT_FOUND = "user_not_found"
    EMAIL_ALREADY_EXISTS = "email_already_exists"
    PROVIDER_NOT_FOUND = "provider_not_found"
    NOT_AUTHENTICATED = "not_authenticated"
    SYSTEM_ALREADY_INITIALIZED = "system_already_initialized"


class TokenError(StrEnum):
    """JWT 解码失败原因的完整列表。"""

    EXPIRED = "expired"
    INVALID_SIGNATURE = "invalid_signature"
    MALFORMED = "malformed"


class AuthErrorResponse(BaseModel):
    """结构化错误响应——取代裸的 `detail` 字符串。"""

    code: AuthErrorCode
    message: str


def token_error_to_code(err: TokenError) -> AuthErrorCode:
    """将 TokenError 映射为 AuthErrorCode——唯一的映射来源。"""
    if err == TokenError.EXPIRED:
        return AuthErrorCode.TOKEN_EXPIRED
    return AuthErrorCode.TOKEN_INVALID
