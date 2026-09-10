"""带版本化哈希格式的密码哈希工具。

哈希格式:``$dfv<N>$<bcrypt_hash>``,其中 ``<N>`` 为版本号。

- **v1**(旧版):``bcrypt(password)`` —— 纯 bcrypt,存在 72 字节
  静默截断问题。
- **v2**(当前):``bcrypt(b64(sha256(password)))`` —— SHA-256 预哈希
  避免了 72 字节截断限制,使完整密码都参与哈希计算。

校验时自动检测版本,对没有前缀的哈希回退为 v1 处理,因此已有部署
会在下次登录时透明升级。
"""

import asyncio
import base64
import hashlib

import bcrypt

_CURRENT_VERSION = 2
_PREFIX_V2 = "$dfv2$"
_PREFIX_V1 = "$dfv1$"


def _pre_hash_v2(password: str) -> bytes:
    """SHA-256 预哈希,用于绕过 bcrypt 的 72 字节限制。"""
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str) -> str:
    """对密码做哈希(当前版本:v2 —— SHA-256 + bcrypt)。"""
    raw = bcrypt.hashpw(_pre_hash_v2(password), bcrypt.gensalt()).decode("utf-8")
    return f"{_PREFIX_V2}{raw}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验密码,自动检测哈希版本。

    接受 v2(``$dfv2$…``)、v1(``$dfv1$…``)以及裸 bcrypt 哈希
    (为兼容版本化之前的数据,按 v1 处理)。
    """
    try:
        if hashed_password.startswith(_PREFIX_V2):
            bcrypt_hash = hashed_password[len(_PREFIX_V2) :]
            return bcrypt.checkpw(_pre_hash_v2(plain_password), bcrypt_hash.encode("utf-8"))

        if hashed_password.startswith(_PREFIX_V1):
            bcrypt_hash = hashed_password[len(_PREFIX_V1) :]
        else:
            bcrypt_hash = hashed_password

        return bcrypt.checkpw(plain_password.encode("utf-8"), bcrypt_hash.encode("utf-8"))
    except ValueError:
        # 哈希格式错误或损坏(如 salt 非法)时 bcrypt 会抛出 ValueError。
        # 这里按失败处理,而不是让请求崩溃。
        return False


def needs_rehash(hashed_password: str) -> bool:
    """哈希使用旧版本、需要重新哈希时返回 True。"""
    return not hashed_password.startswith(_PREFIX_V2)


async def hash_password_async(password: str) -> str:
    """使用 bcrypt 对密码做哈希(非阻塞)。

    将阻塞的 bcrypt 操作包装到线程池中,避免哈希密码时阻塞事件循环。
    """
    return await asyncio.to_thread(hash_password, password)


async def verify_password_async(plain_password: str, hashed_password: str) -> bool:
    """校验密码与哈希是否匹配(非阻塞)。

    将阻塞的 bcrypt 操作包装到线程池中,避免校验密码时阻塞事件循环。
    """
    return await asyncio.to_thread(verify_password, plain_password, hashed_password)
