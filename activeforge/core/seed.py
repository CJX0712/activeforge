"""全局确定性种子：唯一入口 set_all(seed)。

numpy / random / torch(可选) 一次设齐，保证 demo 两次运行逐位一致。
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

_SEED: int = 42
_LOCKED: bool = False


def set_all(seed: int = 42) -> int:
    """设置全局确定性种子，返回实际使用的 seed。"""
    global _SEED
    _SEED = int(seed)
    random.seed(_SEED)
    np.random.seed(_SEED)
    try:  # torch 可选，API 漂移安全
        import torch  # type: ignore

        torch.manual_seed(_SEED)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.manual_seed_all(_SEED)
    except Exception:
        pass
    return _SEED


def get_seed() -> int:
    return _SEED


def rng(seed: Optional[int] = None) -> np.random.Generator:
    """返回一个独立、可复现的 Generator（不污染全局状态）。"""
    return np.random.default_rng(_SEED if seed is None else int(seed))
