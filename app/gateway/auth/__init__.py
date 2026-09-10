"""DeerFlow 的认证模块。

本模块提供:
- 基于 JWT 的认证
- 用于扩展认证方式的 Provider Factory 模式
- 面向存储后端(SQLite)的 UserRepository 接口
"""

from app.gateway.auth.config import AuthConfig, get_auth_config, set_auth_config
from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse, TokenError
from app.gateway.auth.jwt import TokenPayload, create_access_token, decode_token
from app.gateway.auth.local_provider import LocalAuthProvider
from app.gateway.auth.models import User, UserResponse
from app.gateway.auth.password import hash_password, verify_password
from app.gateway.auth.providers import AuthProvider
from app.gateway.auth.repositories.base import UserRepository

__all__ = [
    # Config
    "AuthConfig",
    "get_auth_config",
    "set_auth_config",
    # Errors
    "AuthErrorCode",
    "AuthErrorResponse",
    "TokenError",
    # JWT
    "TokenPayload",
    "create_access_token",
    "decode_token",
    # Password
    "hash_password",
    "verify_password",
    # Models
    "User",
    "UserResponse",
    # Providers
    "AuthProvider",
    "LocalAuthProvider",
    # Repository
    "UserRepository",
]
