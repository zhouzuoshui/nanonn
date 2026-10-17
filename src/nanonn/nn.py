"""基于 CuPy 手写的神经网络层与模块系统。

只实现 ResNet18 所需的最小集合：

- :class:`Module`       模块基类（参数注册、train/eval、state_dict）；
- :class:`Parameter`    可训练参数；
- :class:`Linear`       全连接层；
- :class:`Conv2d`       二维卷积（无 bias 时性能更好，ResNet 中 BN 会接管 bias）；
- :class:`BatchNorm2d`  批归一化（含 running_mean / running_var）；
- :class:`ReLU` / :class:`MaxPool2d` / :class:`AdaptiveAvgPool2d`；
- :class:`Sequential`   顺序容器。

所有权重初始化都使用 Kaiming 正态初始化（fan_in 模式），与 ResNet 论文一致。
"""

from __future__ import annotations

import math
from typing import Iterator

import cupy as cp

from . import functional as F
from .tensor import Tensor

__all__ = [
    "Module",
    "Parameter",
    "Linear",
    "Conv2d",
    "BatchNorm2d",
    "ReLU",
    "MaxPool2d",
    "AdaptiveAvgPool2d",
    "Flatten",
    "Sequential",
    "kaiming_normal_",
    "constant_",
]


class Parameter(Tensor):
    """可训练参数：本质上是 ``requires_grad=True`` 的叶子张量。"""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    """所有网络模块的基类。

    通过 ``__setattr__`` 自动把 :class:`Parameter` 和子 :class:`Module`
    注册到对应的字典中，因此定义网络时可以直接写::

        self.conv = Conv2d(3, 64, 3)
    """

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "training", True)

    # ------------------------------------------------------------------
    # 属性注册
    # ------------------------------------------------------------------
    def __setattr__(self, name: str, value) -> None:
        params = self.__dict__.get("_parameters")
        if params is not None:
            # 重新赋值时先移除旧注册，避免残留
            params.pop(name, None)
            if isinstance(value, Parameter):
                params[name] = value

            modules = self.__dict__.get("_modules")
            modules.pop(name, None)
            if isinstance(value, Module):
                modules[name] = value

        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, value) -> None:
        """注册不参与求导、但需要随模型保存的张量（如 BN 的 running stats）。"""
        self._buffers[name] = value
        object.__setattr__(self, name, value)

    def add_module(self, name: str, module: "Module") -> None:
        self._modules[name] = module
        object.__setattr__(self, name, module)

    # ------------------------------------------------------------------
    # 遍历与状态
    # ------------------------------------------------------------------
    def _named_members(self, kind: str, prefix: str = "", memo: set | None = None):
        """递归遍历参数/缓冲区，并用共享的 memo 去重。

        ``kind`` 取 ``"_parameters"`` 或 ``"_buffers"``。这里显式传字段名
        而不是传闭包，否则递归到子模块时闭包仍然绑定在根模块上。
        """
        if memo is None:
            memo = set()
        for name, module in self._modules.items():
            yield from module._named_members(kind, prefix + name + ".", memo)
        for name, value in getattr(self, kind).items():
            if value is None or id(value) in memo:
                continue
            memo.add(id(value))
            yield prefix + name, value

    def named_parameters(self, prefix: str = ""):
        return self._named_members("_parameters", prefix)

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_buffers(self, prefix: str = ""):
        return self._named_members("_buffers", prefix)

    def buffers(self):
        for _, b in self.named_buffers():
            yield b

    def state_dict(self) -> dict[str, cp.ndarray]:
        """返回 {名称: cupy 数组}，参数与 running stats 都会被包含。"""
        state: dict[str, cp.ndarray] = {}
        for name, p in self.named_parameters():
            state[name] = p.data
        for name, b in self.named_buffers():
            state[name] = b
        return state

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        """把外部状态（numpy / cupy 数组）拷贝进当前模块。

        使用 ``p.data[...] = ...`` 原地写入，保证优化器中持有的参数引用不变。
        """
        own = self.state_dict()
        missing = [k for k in own if k not in state]
        unexpected = [k for k in state if k not in own]
        if strict and (missing or unexpected):
            raise RuntimeError(
                f"state_dict 不匹配: missing={missing}, unexpected={unexpected}"
            )

        for name, target in own.items():
            if name not in state:
                continue
            src = state[name]
            src = cp.asarray(src, dtype=target.dtype)
            if src.shape != target.shape:
                raise RuntimeError(
                    f"参数 {name} 形状不匹配: {src.shape} vs {target.shape}"
                )
            target[...] = src

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = None

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def modules(self):
        """深度优先遍历自身及所有子模块。"""
        yield self
        for m in self._modules.values():
            yield from m.modules()

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for name, module in self._modules.items():
            sub = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {sub}")
        lines.append(")")
        return "\n".join(lines)


# ======================================================================
# 参数初始化
# ======================================================================
def kaiming_normal_(tensor: Tensor, fan_in: int, a: float = 0.0) -> None:
    """Kaiming 正态初始化：std = sqrt(2 / ((1 + a^2) * fan_in))。"""
    std = math.sqrt(2.0 / (1.0 + a * a) / max(fan_in, 1))
    tensor.data[...] = cp.random.normal(
        0.0, std, size=tensor.shape, dtype=tensor.dtype
    )


def constant_(tensor: Tensor, value: float) -> None:
    tensor.data[...] = value


# ======================================================================
# 常用层
# ======================================================================
class Linear(Module):
    """全连接层：y = x @ W^T + b。"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            cp.empty((out_features, in_features), dtype=cp.float32)
        )
        self.bias = (
            Parameter(cp.zeros(out_features, dtype=cp.float32)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        kaiming_normal_(self.weight, self.in_features)
        if self.bias is not None:
            constant_(self.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        # 支持 (N, ..., in_features)，先展平成二维
        if x.ndim > 2:
            x = F.reshape(x, (x.shape[0], -1))
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class Conv2d(Module):
    """二维卷积层，权重形状为 (out_channels, in_channels, kh, kw)。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = Parameter(
            cp.empty(
                (out_channels, in_channels, kernel_size, kernel_size),
                dtype=cp.float32,
            )
        )
        self.bias = (
            Parameter(cp.zeros(out_channels, dtype=cp.float32)) if bias else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        kaiming_normal_(self.weight, fan_in)
        if self.bias is not None:
            constant_(self.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        out = F.conv2d(
            x, self.weight, stride=self.stride, padding=self.padding
        )
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


class BatchNorm2d(Module):
    """二维批归一化。

    训练时用当前 batch 的统计量，并动量更新 running stats；
    评估时使用 running stats（``Module.eval()`` 切换）。
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = Parameter(cp.ones(num_features, dtype=cp.float32))
        self.bias = Parameter(cp.zeros(num_features, dtype=cp.float32))
        self.register_buffer("running_mean", cp.zeros(num_features, dtype=cp.float32))
        self.register_buffer("running_var", cp.ones(num_features, dtype=cp.float32))

    def forward(self, x: Tensor) -> Tensor:
        return F.batch_norm(
            x,
            self.weight,
            self.bias,
            self.running_mean,
            self.running_var,
            eps=self.eps,
            momentum=self.momentum,
            training=self.training,
        )


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int = 2, stride: int | None = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def forward(self, x: Tensor) -> Tensor:
        return F.max_pool2d(x, self.kernel_size, self.stride)


class AdaptiveAvgPool2d(Module):
    """全局平均池化（当前只支持输出 1x1）。"""

    def __init__(self, output_size: int = 1):
        super().__init__()
        if output_size != 1:
            raise NotImplementedError("AdaptiveAvgPool2d 目前只支持 output_size=1")
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return F.mean(x, axis=(2, 3))


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.reshape(x, (x.shape[0], -1))


class Sequential(Module):
    """按顺序执行子模块的容器。"""

    def __init__(self, *modules: Module):
        super().__init__()
        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self) -> int:
        return len(self._modules)

    def __getitem__(self, index: int) -> Module:
        return list(self._modules.values())[index]
