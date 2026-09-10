"""用于抽象数据库操作的用户仓储接口。"""

from abc import ABC, abstractmethod

from app.gateway.auth.models import User


class UserNotFoundError(LookupError):
    """用户仓储操作针对不存在的行时抛出。

    继承自 :class:`LookupError`,已捕获 ``LookupError`` 处理"实体缺失"
    的调用方无需改动;特定调用点也可以单独捕获本类,以区分"更新期间
    被并发删除"与其他查询缺失。
    """


class UserRepository(ABC):
    """用户数据存储的抽象接口。

    实现该接口以支持不同的存储后端(SQLite)。
    """

    @abstractmethod
    async def create_user(self, user: User) -> User:
        """创建新用户。

        Args:
            user: 要创建的 User 对象

        Returns:
            创建后已分配 ID 的 User

        Raises:
            ValueError: 邮箱已存在时抛出
        """
        raise NotImplementedError

    @abstractmethod
    async def get_user_by_id(self, user_id: str) -> User | None:
        """按 ID 获取用户。

        Args:
            user_id: 用户 UUID 字符串

        Returns:
            找到返回 User,否则返回 None
        """
        raise NotImplementedError

    @abstractmethod
    async def get_user_by_email(self, email: str) -> User | None:
        """按邮箱获取用户。

        Args:
            email: 用户邮箱地址

        Returns:
            找到返回 User,否则返回 None
        """
        raise NotImplementedError

    @abstractmethod
    async def update_user(self, user: User) -> User:
        """更新已有用户。

        Args:
            user: 字段已更新的 User 对象

        Returns:
            更新后的 User

        Raises:
            UserNotFoundError: ``user.id`` 对应的行不存在时抛出。这是
                硬性失败(而非静默跳过),避免调用方把并发删除竞争误当
                作更新成功。
        """
        raise NotImplementedError

    @abstractmethod
    async def count_users(self) -> int:
        """返回注册用户总数。"""
        raise NotImplementedError

    @abstractmethod
    async def count_admin_users(self) -> int:
        """返回 system_role == 'admin' 的用户数。"""
        raise NotImplementedError

    @abstractmethod
    async def get_user_by_oauth(self, provider: str, oauth_id: str) -> User | None:
        """按 OAuth 提供者和 ID 获取用户。

        Args:
            provider: OAuth 提供者名称(如 'github'、'google')
            oauth_id: OAuth 提供者返回的用户 ID

        Returns:
            找到返回 User,否则返回 None
        """
        raise NotImplementedError
