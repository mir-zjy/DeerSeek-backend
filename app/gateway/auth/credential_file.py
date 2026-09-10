"""将初始管理员凭据写入受限文件,而不是输出到日志。

把密钥输出到 stdout/stderr 是众所周知的 CodeQL 告警
(py/clear-text-logging-sensitive-data)——生产环境中这些日志会被
采集到 ELK/Splunk 等系统,成为密钥扩散的源头。本辅助函数将凭据写入
仅进程用户可读的 0600 文件,并返回路径,让调用方记录**路径**(而非
密码)供运维人员查看。
"""

from __future__ import annotations

import os
from pathlib import Path

from deerflow.config.paths import get_paths

_CREDENTIAL_FILENAME = "admin_initial_credentials.txt"


def write_initial_credentials(email: str, password: str, *, label: str = "initial") -> Path:
    """将管理员邮箱和密码写入 ``{base_dir}/admin_initial_credentials.txt``。

    通过 ``os.open`` 以 0600 模式**原子地**创建文件,即使在
    ``write_text`` 与 ``chmod`` 之间的单个系统调用窗口内,密码也绝不
    会被其他用户读取。

    ``label`` 在文件头中区分 "initial"(首次创建)和 "reset"(密码重置),
    使运维人员在重启后拿到文件时能分辨它是由哪个事件生成的。

    返回文件的绝对 :class:`Path`。
    """
    target = get_paths().base_dir / _CREDENTIAL_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    content = (
        f"# DeerFlow admin {label} credentials\n# This file is generated on first boot or password reset.\n# Change the password after login via Settings -> Account,\n# then delete this file.\n#\nemail: {email}\npassword: {password}\n"
    )

    # 原子地以 0600 创建或截断。使用 O_TRUNC(而非 O_EXCL),使重置密码
    # 路径可以直接重写已有文件,无需先删除再创建。
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(content)

    return target.resolve()
