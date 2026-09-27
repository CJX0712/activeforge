"""配置：支持 ENV_ACTIVEFORGE_* 环境变量覆盖。"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvConfig:
    seed: int
    n_jobs: int

    @staticmethod
    def load() -> "EnvConfig":
        return EnvConfig(
            seed=int(os.environ.get("ACTIVEFORGE_SEED", "42")),
            n_jobs=int(os.environ.get("ACTIVEFORGE_N_JOBS", "1")),
        )

    def as_dict(self) -> dict:
        return {"seed": self.seed, "n_jobs": self.n_jobs}
