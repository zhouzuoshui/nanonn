"""基于 CuPy 的极简自动求导引擎。

本模块只依赖 CuPy（不依赖 PyTorch），实现了训练 ResNet18 所需的全部
自动微分基础设施：

- :class:`Tensor`   张量对象，持有数据、梯度以及计算图节点（``grad_fn``）；
- :class:`Function` 算子基类，子类通过 ``forward`` / ``backward`` 描述一个
  可微算子，调用 ``Function.apply(...)`` 即可自动构建计算图；
- :func:`no_grad`   上下文管理器，用于推理/评估阶段关闭计算图构建；
- ``Tensor.backward()`` 按拓扑序执行反向传播。

设计取舍：
1. 反向图采用“动态图 + 闭包”方式，与 PyTorch 的 eager 模式一致；
2. 中间张量的梯度在反向传播使用完后立即释放，避免显存峰值过高；
3. 所有计算都在 GPU 上完成，CPU 只负责数据加载与日志。
"""

from __future__ import annotations

import contextlib
from typing import Any, Sequence

import cupy as cp

__all__ = [
    "Tensor",
    "Function",
    "no_grad",
    "is_grad_enabled",
    "as_tensor",
]

# 全局梯度开关：no_grad 将其置为 False，从而跳过计算图构建。
_grad_enabled = True


def is_grad_enabled() -> bool:
    """返回当前是否处于梯度计算模式。"""
    return _grad_enabled


@contextlib.contextmanager
def no_grad():
    """关闭自动求导的上下文管理器。

    用法::

        with no_grad():
            logits = model(x)   # 不会构建计算图，也不会保存中间激活值
    """
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old


def as_tensor(data: Any) -> "Tensor":
    """把任意数组/标量转换成 :class:`Tensor`（已经是 Tensor 则原样返回）。

    Python 标量统一转成 float32，避免与网络参数（float32）做运算时被
    提升成 float64，从而浪费显存和带宽。
    """
    if isinstance(data, Tensor):
        return data
    if isinstance(data, (bool, int, float)):
        return Tensor(cp.asarray(data, dtype=cp.float32))
    return Tensor(data)


class GradFn:
    """计算图中的一个节点，记录“产生该张量所用的算子及输入”。

    ``backward`` 会调用算子类中定义的静态反向函数，返回每个输入的梯度。
    """

    __slots__ = ("fn", "ctx", "inputs")

    def __init__(self, fn: type["Function"], ctx: dict, inputs: Sequence["Tensor"]):
        self.fn = fn
        self.ctx = ctx
        self.inputs = inputs

    def backward(self, grad: cp.ndarray):
        return self.fn.backward(self.ctx, grad)


class Tensor:
    """GPU 张量，支持自动求导。

    属性
    ----
    data:
        实际的 :class:`cupy.ndarray`。所有算子都直接读写它。
    requires_grad:
        是否需要对该张量求梯度。网络参数为 True，数据为 False。
    grad:
        损失对该张量的梯度，形状与 ``data`` 相同；反向传播时累加。
    grad_fn:
        指向产生该张量的 :class:`GradFn`；叶子节点为 None。
    is_leaf:
        是否为叶子节点（用户直接创建、没有计算图来源的张量）。
    """

    __slots__ = ("data", "requires_grad", "grad", "grad_fn", "is_leaf")

    def __init__(self, data: Any, requires_grad: bool = False):
        if not isinstance(data, cp.ndarray):
            data = cp.asarray(data)
        self.data = data
        # 在 no_grad 上下文里创建张量时强制关闭梯度，避免意外构图。
        self.requires_grad = bool(requires_grad) and _grad_enabled
        self.grad: cp.ndarray | None = None
        self.grad_fn: GradFn | None = None
        self.is_leaf = True

    # ------------------------------------------------------------------
    # 基本属性
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self) -> int:
        return int(self.data.size)

    @property
    def T(self) -> "Tensor":
        """二维转置（仅为方便使用，内部走统一的 transpose 算子）。"""
        from . import functional as F

        return F.transpose(self, (1, 0))

    def __repr__(self) -> str:
        return (
            f"Tensor(shape={self.shape}, dtype={self.dtype}, "
            f"requires_grad={self.requires_grad})"
        )

    # ------------------------------------------------------------------
    # 反向传播
    # ------------------------------------------------------------------
    def backward(self, grad: cp.ndarray | None = None) -> None:
        """从当前张量出发执行反向传播。

        参数
        ----
        grad:
            上游梯度。若为 None 则要求当前张量是标量（损失），默认取全 1。
        """
        if not self.requires_grad:
            raise RuntimeError("该张量不需要梯度，无法调用 backward()")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "非标量张量调用 backward() 时必须显式提供 grad"
                )
            grad = cp.ones_like(self.data)
        elif not isinstance(grad, cp.ndarray):
            grad = cp.asarray(grad, dtype=self.data.dtype)

        # ---- 1. 迭代式后序遍历，得到拓扑序（输入在前，输出在后） ----
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            fn = node.grad_fn
            if fn is not None:
                for inp in fn.inputs:
                    if inp.requires_grad and id(inp) not in visited:
                        stack.append((inp, False))

        # ---- 2. 按逆拓扑序（输出 -> 输入）传播梯度 ----
        self.grad = grad if self.grad is None else self.grad + grad
        for node in reversed(topo):
            g = node.grad
            if g is None:
                continue
            fn = node.grad_fn
            if fn is None:  # 叶子节点，梯度保留给优化器使用
                continue

            in_grads = fn.backward(g)
            for inp, ig in zip(fn.inputs, in_grads):
                if ig is None or not inp.requires_grad:
                    continue
                inp.grad = ig if inp.grad is None else inp.grad + ig

            # 非叶子节点：传播完毕后立刻释放，显著降低显存占用。
            if not node.is_leaf:
                node.grad = None

    # ------------------------------------------------------------------
    # 常用工具
    # ------------------------------------------------------------------
    def zero_grad(self) -> None:
        self.grad = None

    def detach(self) -> "Tensor":
        """返回共享底层数据但不参与求导的新张量。"""
        return Tensor(self.data, requires_grad=False)

    def item(self) -> float:
        return float(self.data.item())

    def numpy(self):
        """拷贝回 CPU（numpy.ndarray）。"""
        return cp.asnumpy(self.data)

    def astype(self, dtype) -> "Tensor":
        """类型转换（不参与求导，主要用于日志/可视化）。"""
        return Tensor(self.data.astype(dtype), requires_grad=False)

    # ------------------------------------------------------------------
    # 便捷方法：延迟导入 functional，避免循环依赖
    # ------------------------------------------------------------------
    def reshape(self, *shape) -> "Tensor":
        from . import functional as F

        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return F.reshape(self, shape)

    def transpose(self, *axes) -> "Tensor":
        from . import functional as F

        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return F.transpose(self, axes)

    def sum(self, axis=None, keepdims: bool = False) -> "Tensor":
        from . import functional as F

        return F.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims: bool = False) -> "Tensor":
        from . import functional as F

        return F.mean(self, axis=axis, keepdims=keepdims)

    def relu(self) -> "Tensor":
        from . import functional as F

        return F.relu(self)

    # ------------------------------------------------------------------
    # 运算符重载
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        from . import functional as F

        return F.getitem(self, key)

    def __add__(self, other):
        from . import functional as F

        return F.add(self, other)

    def __radd__(self, other):
        from . import functional as F

        return F.add(other, self)

    def __sub__(self, other):
        from . import functional as F

        return F.sub(self, other)

    def __rsub__(self, other):
        from . import functional as F

        return F.sub(other, self)

    def __mul__(self, other):
        from . import functional as F

        return F.mul(self, other)

    def __rmul__(self, other):
        from . import functional as F

        return F.mul(other, self)

    def __truediv__(self, other):
        from . import functional as F

        return F.div(self, other)

    def __rtruediv__(self, other):
        from . import functional as F

        return F.div(other, self)

    def __matmul__(self, other):
        from . import functional as F

        return F.matmul(self, other)

    def __rmatmul__(self, other):
        from . import functional as F

        return F.matmul(other, self)

    def __neg__(self):
        from . import functional as F

        return F.neg(self)

    def __pow__(self, exponent):
        from . import functional as F

        return F.pow(self, exponent)


def _unbroadcast(grad: cp.ndarray, shape: tuple[int, ...]) -> cp.ndarray:
    """把广播后的梯度规约回原始形状。

    例如 ``(N, C, H, W) + (C, 1, 1)`` 的反向传播中，后者的梯度需要
    在 batch 和空间维度上求和。
    """
    if grad.shape == shape:
        return grad
    # 1) 去掉多出来的前导维度
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) 对原始形状中大小为 1 的维度做 keepdims 求和
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Function:
    """可微算子基类。

    子类需要实现两个静态方法::

        @staticmethod
        def forward(ctx, *inputs): ...
            # ctx 是普通 dict，可用于保存反向传播所需的信息

        @staticmethod
        def backward(ctx, grad_output): ...
            # 返回与 forward 的 Tensor 输入一一对应的梯度元组

    非张量参数（stride、padding、axis 等）通过关键字参数传入，
    会被存进 ``ctx``，既不会参与求导，也不会被当成计算图输入。
    """

    @classmethod
    def apply(cls, *inputs, **params) -> Tensor:
        tensors = tuple(as_tensor(x) for x in inputs)
        ctx: dict[str, Any] = dict(params)

        data = cls.forward(ctx, *(t.data for t in tensors))

        requires_grad = any(t.requires_grad for t in tensors)
        out = Tensor(data, requires_grad=requires_grad)
        if requires_grad and _grad_enabled:
            out.is_leaf = False
            out.grad_fn = GradFn(cls, ctx, tensors)
        return out
