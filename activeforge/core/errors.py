"""错误码 E100~E500，按模块分区，便于定位。"""

from __future__ import annotations


class ActiveForgeError(Exception):
    code = "E000"
    """基类错误。"""

    def __init__(self, msg: str = ""):
        super().__init__(f"[{self.code}] {msg}")


class ConfigError(ActiveForgeError):
    code = "E100"


class DataError(ActiveForgeError):
    code = "E200"


class LearnerError(ActiveForgeError):
    code = "E300"


class StrategyError(ActiveForgeError):
    code = "E400"


class PipelineError(ActiveForgeError):
    code = "E500"
