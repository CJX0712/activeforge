"""查询策略（主动学习核心）。

策略谱系（覆盖 AL 经典 SOTA 分类）：
  - RandomStrategy         : 随机采样（强基线 / 对照）
  - UncertaintyStrategy     : 不确定性采样（least_confident / margin / entropy）—— Lewis & Catlett 1994, Settles 2009
  - CommitteeStrategy       : Query-By-Committee + BALD 近似（Seung 1992; Houlsby et al. 2011）
  - CoresetStrategy         : 核心集 / k-center 多样性采样（Sener & Savare 2018）
  - HybridStrategy          : 批感知「不确定性 × 多样性」混合（BatchBALD / BADGE 思路，本系统创新点）

分数语义：越大越值得查询（越不确定 / 越具多样性）。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _entropy(proba: np.ndarray) -> np.ndarray:
    """逐行香农熵（bit），proba: (U, C)。"""
    p = np.clip(proba, 1e-12, 1.0)
    return -np.sum(p * np.log2(p), axis=1)


def _standardized(pool) -> np.ndarray:
    return StandardScaler().fit(pool.X).transform(pool.X)


class RandomStrategy:
    name = "random"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, pool, learner, n: int) -> np.ndarray:
        ux = pool.unlabeled_global
        rng = np.random.default_rng(self.seed + pool.n_labeled)
        return rng.choice(ux, size=min(n, len(ux)), replace=False)


class UncertaintyStrategy:
    name = "uncertainty"

    def __init__(self, mode: str = "entropy"):
        if mode not in ("least_confident", "margin", "entropy"):
            raise ValueError(f"未知 uncertainty 模式: {mode}")
        self.mode = mode

    def select(self, pool, learner, n: int) -> np.ndarray:
        ux = pool.unlabeled_global
        proba = learner.predict_proba(pool.unlabeled_X)  # (U, C)
        if proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])
        ordered = np.sort(proba, axis=1)
        max1 = ordered[:, -1]
        max2 = ordered[:, -2]
        if self.mode == "least_confident":
            score = 1.0 - max1
        elif self.mode == "margin":
            score = max2 - max1  # 越小 margin → 越大 score
        else:  # entropy
            score = _entropy(proba)
        k = min(n, len(ux))
        return ux[np.argsort(-score)[:k]]


class CommitteeStrategy:
    name = "committee"

    def __init__(self, n_members: int = 5, mode: str = "bald", seed: int = 42):
        if mode not in ("bald", "vote_entropy"):
            raise ValueError(f"未知 committee 模式: {mode}")
        self.n_members = n_members
        self.mode = mode
        self.seed = seed

    def select(self, pool, learner, n: int) -> np.ndarray:
        ux = pool.unlabeled_global
        U = len(ux)
        k = min(n, U)
        if U == 0:
            return ux[:0]
        rng = np.random.default_rng(self.seed + pool.n_labeled)
        LX, LY = pool.labeled_X, pool.labeled_y
        global_classes = np.unique(LY)
        members = []
        for m in range(self.n_members):
            # 保证每个成员都见到全部类别（避免 bootstrap 漏类导致列数不一致）
            boot = []
            for c in global_classes:
                cidx = np.where(LY == c)[0]
                boot.append(int(rng.choice(cidx, size=1, replace=True)[0]))
            n_fill = len(LX) - len(boot)
            if n_fill > 0:
                boot += rng.choice(len(LX), size=n_fill, replace=True).tolist()
            dt = DecisionTreeClassifier(
                max_depth=6, random_state=int(self.seed + m + pool.n_labeled)
            )
            dt.fit(LX[boot], LY[boot])
            members.append(dt)
        # 对齐到全局类别列（缺类成员对应列补 0，保持形状一致）
        probas_aligned = []
        for mbr in members:
            p = mbr.predict_proba(pool.unlabeled_X)
            full = np.zeros((p.shape[0], len(global_classes)), dtype=float)
            mclasses = mbr.classes_
            for i, c in enumerate(mclasses):
                j = int(np.where(global_classes == c)[0][0])
                full[:, j] = p[:, i]
            probas_aligned.append(full)
        probas = np.stack(probas_aligned, axis=0)
        C = probas.shape[2]
        mean_proba = probas.mean(axis=0)  # (U, C)
        if self.mode == "bald":
            h_mean = _entropy(mean_proba)  # (U,)
            h_each = np.stack([_entropy(probas[m]) for m in range(self.n_members)])  # (M, U)
            score = h_mean - h_each.mean(axis=0)  # (U,)
        else:  # vote_entropy
            votes = np.stack([np.argmax(probas[m], axis=1) for m in range(self.n_members)], axis=0)
            counts = np.apply_along_axis(
                lambda col: np.bincount(col, minlength=C), axis=0, arr=votes
            )  # (C, U)
            vote_dist = counts / self.n_members
            score = _entropy(vote_dist.T)
        return ux[np.argsort(-score)[:k]]


class CoresetStrategy:
    name = "coreset"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, pool, learner, n: int) -> np.ndarray:
        ux = pool.unlabeled_global
        k = min(n, len(ux))
        if k == 0:
            return ux[:0]
        Xs = _standardized(pool)
        S = list(pool.labeled_mask.nonzero()[0].tolist())
        # 距离矩阵：未标注点到已选集的最小 L2（标准化坐标）
        unlab_coords = Xs[ux]
        sel_coords = Xs[S]
        min_dist = np.min(
            np.linalg.norm(unlab_coords[:, None, :] - sel_coords[None, :, :], axis=2),
            axis=1,
        )
        chosen: List[int] = []
        for _ in range(k):
            j = int(np.argmax(min_dist))
            gidx = int(ux[j])
            chosen.append(gidx)
            # 更新：把新选点并入已选集，重算最小距离
            new_coord = Xs[gidx][None, :]
            d_new = np.linalg.norm(unlab_coords - new_coord, axis=1)
            min_dist = np.minimum(min_dist, d_new)
            min_dist[j] = -1.0  # 已选，屏蔽
        return np.array(chosen, dtype=int)


class HybridStrategy:
    """批感知混合策略（创新点）：不确定性 × 批内多样性。

    逐点贪心：score = u_norm * d_norm，其中 d_norm 为该点到
    (已标注 ∪ 本批已选) 的最小标准化距离（多样性）。模拟 BatchBALD / BADGE
    的「既不确定又互补」选点思想，避免纯 top-k 不确定性的冗余查询。
    """

    name = "hybrid"

    def __init__(self, seed: int = 42):
        self.seed = seed

    def select(self, pool, learner, n: int) -> np.ndarray:
        ux = pool.unlabeled_global
        U = len(ux)
        k = min(n, U)
        if k == 0:
            return ux[:0]
        Xs = _standardized(pool)
        unlab_coords = Xs[ux]
        # 覆盖集 = 已标注点
        covered = list(pool.labeled_mask.nonzero()[0].tolist())
        covered_coords = Xs[covered] if covered else np.empty((0, Xs.shape[1]))
        proba = learner.predict_proba(pool.unlabeled_X)
        if proba.shape[1] == 1:
            proba = np.hstack([1 - proba, proba])
        u = _entropy(proba)
        u_norm = (u - u.min()) / (u.max() - u.min() + 1e-12)
        chosen: List[int] = []
        # 当前覆盖坐标（含本批已选），动态增长；已选点 d_norm=0 → score=0 自然不再被选中
        cov = covered_coords.copy()
        for _ in range(k):
            if cov.shape[0] == 0:
                d_norm = np.ones(U)
            else:
                d = np.linalg.norm(unlab_coords[:, None, :] - cov[None, :, :], axis=2)
                d_norm = np.min(d, axis=1)
            d_norm = (d_norm - d_norm.min()) / (d_norm.max() - d_norm.min() + 1e-12)
            score = u_norm * d_norm
            j = int(np.argmax(score))
            gidx = int(ux[j])
            chosen.append(gidx)
            cov = np.vstack([cov, Xs[gidx][None, :]])
        return np.array(chosen, dtype=int)


# 策略工厂
def build_strategy(name: str, **kw):
    name = name.lower()
    if name == "random":
        return RandomStrategy(seed=kw.get("seed", 42))
    if name == "uncertainty":
        return UncertaintyStrategy(mode=kw.get("mode", "entropy"))
    if name == "committee":
        return CommitteeStrategy(
            n_members=kw.get("n_members", 5), mode=kw.get("mode", "bald"),
            seed=kw.get("seed", 42),
        )
    if name == "coreset":
        return CoresetStrategy(seed=kw.get("seed", 42))
    if name == "hybrid":
        return HybridStrategy(seed=kw.get("seed", 42))
    raise ValueError(f"未知策略: {name}")


STRATEGY_NAMES = ["random", "uncertainty", "committee", "coreset", "hybrid"]
