"""重置管理员密码的 CLI 工具。

用法:
    python -m app.gateway.auth.reset_admin
    python -m app.gateway.auth.reset_admin --email admin@example.com

新密码写入 ``.deer-flow/admin_initial_credentials.txt``(0600 模式)
而不是打印输出,因此 CI / 日志采集系统永远不会看到明文密钥。
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
import sys

from sqlalchemy import select

from app.gateway.auth.credential_file import write_initial_credentials
from app.gateway.auth.password import hash_password
from app.gateway.auth.repositories.sqlite import SQLiteUserRepository
from deerflow.persistence.user.model import UserRow


async def _run(email: str | None) -> int:
    from deerflow.config import get_app_config
    from deerflow.persistence.engine import (
        close_engine,
        get_session_factory,
        init_engine_from_config,
    )

    config = get_app_config()
    await init_engine_from_config(config.database)
    try:
        sf = get_session_factory()
        if sf is None:
            print("Error: persistence engine not available (check config.database).", file=sys.stderr)
            return 1

        repo = SQLiteUserRepository(sf)

        if email:
            user = await repo.get_user_by_email(email)
        else:
            # 通过直接 SELECT 查找第一个管理员——仓储层没有暴露
            # "第一个管理员"的辅助方法,我们也不想仅为这个 CLI
            # 专门添加一个。
            async with sf() as session:
                stmt = select(UserRow).where(UserRow.system_role == "admin").limit(1)
                row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                user = None
            else:
                user = await repo.get_user_by_id(row.id)

        if user is None:
            if email:
                print(f"Error: user '{email}' not found.", file=sys.stderr)
            else:
                print("Error: no admin user found.", file=sys.stderr)
            return 1

        new_password = secrets.token_urlsafe(16)
        user.password_hash = hash_password(new_password)
        user.token_version += 1
        user.needs_setup = True
        await repo.update_user(user)

        cred_path = write_initial_credentials(user.email, new_password, label="reset")
        print(f"Password reset for: {user.email}")
        print(f"Credentials written to: {cred_path} (mode 0600)")
        print("Next login will require setup (new email + password).")
        return 0
    finally:
        await close_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset admin password")
    parser.add_argument("--email", help="Admin email (default: first admin found)")
    args = parser.parse_args()

    exit_code = asyncio.run(_run(args.email))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
