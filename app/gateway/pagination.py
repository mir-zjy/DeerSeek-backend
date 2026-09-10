"""Gateway 路由共享的分页辅助函数。"""

from __future__ import annotations


def trim_run_message_page(rows: list[dict], *, limit: int, after_seq: int | None) -> tuple[list[dict], bool]:
    """裁剪 ``limit + 1`` 大小的 run 消息页,同时保持分页边界正确。"""
    has_more = len(rows) > limit
    if not has_more:
        return rows, False

    if after_seq is not None:
        return rows[:limit], True

    return rows[-limit:], True
