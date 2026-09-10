"""Gateway 层的共享工具函数。"""


def sanitize_log_param(value: str) -> str:
    """去除控制字符,防止日志注入。"""
    return value.replace("\n", "").replace("\r", "").replace("\x00", "")
