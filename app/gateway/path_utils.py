"""thread 虚拟路径(如 mnt/user-data/outputs/...)的共享路径解析。"""

from pathlib import Path

from fastapi import HTTPException

from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id


def resolve_thread_virtual_path(thread_id: str, virtual_path: str) -> Path:
    """将虚拟路径解析为 thread user-data 下的实际文件系统路径。

    Args:
        thread_id: thread ID。
        virtual_path: 沙箱内看到的虚拟路径
                      (如 /mnt/user-data/outputs/file.txt)。

    Returns:
        解析后的文件系统路径。

    Raises:
        HTTPException: 路径非法或超出允许目录时抛出。
    """
    try:
        return get_paths().resolve_virtual_path(thread_id, virtual_path, user_id=get_effective_user_id())
    except ValueError as e:
        status = 403 if "traversal" in str(e) else 400
        raise HTTPException(status_code=status, detail=str(e))
