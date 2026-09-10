"""DeerFlow 的认证配置。"""

import logging
import os
import secrets

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SECRET_FILE = ".jwt_secret"


class AuthConfig(BaseModel):
    """JWT 及认证相关配置。启动时解析一次。

    注意:``users`` 表现在位于由 ``deerflow.persistence.engine`` 管理的
    共享持久化数据库中。旧的 ``users_db_path`` 配置项已移除——用户存储
    与其他表一样通过 ``config.database`` 配置。
    """

    jwt_secret: str = Field(
        ...,
        description="Secret key for JWT signing. MUST be set via AUTH_JWT_SECRET.",
    )
    token_expiry_days: int = Field(default=7, ge=1, le=30)
    oauth_github_client_id: str | None = Field(default=None)
    oauth_github_client_secret: str | None = Field(default=None)


_auth_config: AuthConfig | None = None


def _load_or_create_secret() -> str:
    """从 ``{base_dir}/.jwt_secret`` 加载持久化的 JWT 密钥,不存在则生成并持久化新密钥。"""
    from deerflow.config.paths import get_paths

    paths = get_paths()
    secret_file = paths.base_dir / _SECRET_FILE

    try:
        if secret_file.exists():
            secret = secret_file.read_text(encoding="utf-8").strip()
            if secret:
                return secret
    except OSError as exc:
        raise RuntimeError(f"Failed to read JWT secret from {secret_file}. Set AUTH_JWT_SECRET explicitly or fix DEER_FLOW_HOME/base directory permissions so DeerFlow can read its persisted auth secret.") from exc

    secret = secrets.token_urlsafe(32)
    try:
        secret_file.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(secret_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(secret)
    except OSError as exc:
        raise RuntimeError(f"Failed to persist JWT secret to {secret_file}. Set AUTH_JWT_SECRET explicitly or fix DEER_FLOW_HOME/base directory permissions so DeerFlow can store a stable auth secret.") from exc
    return secret


def get_auth_config() -> AuthConfig:
    """获取全局 AuthConfig 实例。首次调用时从环境变量解析。"""
    global _auth_config
    if _auth_config is None:
        from dotenv import load_dotenv

        load_dotenv()
        jwt_secret = os.environ.get("AUTH_JWT_SECRET")
        if not jwt_secret:
            jwt_secret = _load_or_create_secret()
            os.environ["AUTH_JWT_SECRET"] = jwt_secret
            logger.warning(
                "⚠ AUTH_JWT_SECRET is not set — using an auto-generated secret "
                "persisted to .jwt_secret. Sessions will survive restarts. "
                "For production, add AUTH_JWT_SECRET to your .env file: "
                'python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )
        _auth_config = AuthConfig(jwt_secret=jwt_secret)
    return _auth_config


def set_auth_config(config: AuthConfig) -> None:
    """设置全局 AuthConfig 实例(用于测试)。"""
    global _auth_config
    _auth_config = config
