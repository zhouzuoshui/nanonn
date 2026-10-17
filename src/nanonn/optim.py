"""基于 CuPy 手写的优化器与学习率调度器。

包含 ResNet 训练常用的：

- :class:`SGD`     带动量 / Nesterov / weight decay 的随机梯度下降；
- :class:`AdamW`   解耦权重衰减的 Adam；
- :class:`CosineAnnealingLR` 带线性 warmup 的余弦退火；
- :func:`clip_grad_norm_`    按全局范数裁剪梯度。

优化器直接原地修改 ``parameter.data``，不经过自动求导图，
因此不需要使用 ``no_grad``。
"""

from __future__ import annotations

import math
from typing import Iterable

import cupy as cp

from .nn import Parameter

__all__ = ["Optimizer", "SGD", "AdamW", "CosineAnnealingLR", "clip_grad_norm_"]


class Optimizer:
    """优化器基类。"""

    def __init__(self, params: Iterable[Parameter], lr: float):
        # 转成 list 固定顺序，保证 state 与参数一一对应
        self.params = list(params)
        self.lr = float(lr)

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:  # pragma: no cover - 由子类实现
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降（可带动量与 Nesterov）。"""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 0.1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super().__init__(params, lr)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.nesterov = bool(nesterov)
        if self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov 动量需要 momentum > 0")

        # 动量缓冲区与参数一一对应（延迟分配，避免无谓显存占用）
        self.buffers: list[cp.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            # L2 正则：直接把 weight decay 加到梯度上
            if self.weight_decay:
                g = g + self.weight_decay * p.data

            if self.momentum > 0:
                buf = self.buffers[i]
                if buf is None:
                    buf = cp.zeros_like(p.data)
                    self.buffers[i] = buf
                buf *= self.momentum
                buf += g
                g = g + self.momentum * buf if self.nesterov else buf

            p.data -= self.lr * g


class AdamW(Optimizer):
    """AdamW：权重衰减与梯度更新解耦。"""

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m: list[cp.ndarray | None] = [None] * len(self.params)
        self.v: list[cp.ndarray | None] = [None] * len(self.params)

    def step(self) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        # 偏差修正系数
        c1 = 1.0 - b1 ** self.t
        c2 = 1.0 - b2 ** self.t

        for i, p in enumerate(self.params):
            g = p.grad
            if g is None:
                continue

            if self.m[i] is None:
                self.m[i] = cp.zeros_like(p.data)
                self.v[i] = cp.zeros_like(p.data)

            m, v = self.m[i], self.v[i]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)

            m_hat = m / c1
            v_hat = v / c2
            update = m_hat / (cp.sqrt(v_hat) + self.eps)

            if self.weight_decay:
                # 解耦权重衰减：直接作用在参数上
                p.data -= self.lr * self.weight_decay * p.data

            p.data -= self.lr * update


class CosineAnnealingLR:
    """带线性 warmup 的余弦退火学习率。

    第 ``epoch`` 个 epoch 的学习率为::

        warmup 阶段:  base_lr * (epoch + 1) / warmup_epochs
        余弦阶段:     min_lr + 0.5 * (base_lr - min_lr)
                      * (1 + cos(pi * progress))
    """

    def __init__(
        self,
        base_lr: float,
        total_epochs: int,
        warmup_epochs: int = 0,
        min_lr: float = 0.0,
    ):
        self.base_lr = float(base_lr)
        self.total_epochs = int(total_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.min_lr = float(min_lr)

    def get_lr(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs

        progress = (epoch - self.warmup_epochs) / max(
            1, self.total_epochs - self.warmup_epochs
        )
        progress = min(max(progress, 0.0), 1.0)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )


def clip_grad_norm_(params: Iterable[Parameter], max_norm: float) -> float:
    """按全局 L2 范数裁剪梯度，返回裁剪前的总范数。"""
    params = [p for p in params if p.grad is not None]
    if not params:
        return 0.0

    total_sq = cp.zeros((), dtype=cp.float64)
    for p in params:
        total_sq += cp.sum(p.grad * p.grad, dtype=cp.float64)
    total_norm = float(cp.sqrt(total_sq))

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale

    return total_norm
