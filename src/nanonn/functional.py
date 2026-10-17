"""基于 CuPy 手写的可微算子集合。

每个算子都是一个 :class:`~nanonn.tensor.Function` 子类：

* ``forward`` 只使用 CuPy 原生 API（矩阵乘、逐元素运算、reshape 等）；
* ``backward`` 手写梯度公式，返回与 forward 的 Tensor 输入一一对应的梯度。

这里实现的是 ResNet18 训练所需的最小算子集合：
加减乘除、矩阵乘、reshape/transpose、sum/mean、ReLU、
2D 卷积、BatchNorm2d、MaxPool2d、交叉熵。

**显存友好的卷积实现**
------------------------
CIFAR-100 的 batch=128、32x32 特征图做 3x3 卷积时，若一次性展开
im2col，单个卷积层就需要数 GB；本实现按 batch 分块（chunk）做
im2col + 矩阵乘，把临时显存控制在 ``_MAX_IM2COL_BYTES`` 以内，
因此可以在 6GB 显存的笔记本 3060 上训练。
"""

from __future__ import annotations

import cupy as cp

from .tensor import Function, Tensor, _unbroadcast, as_tensor

__all__ = [
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "pow",
    "matmul",
    "reshape",
    "transpose",
    "sum",
    "mean",
    "relu",
    "exp",
    "log",
    "conv2d",
    "batch_norm",
    "max_pool2d",
    "cross_entropy",
]

# im2col 每块允许占用的最大字节数（64MB）。分块越大，矩阵乘效率越高；
# 分块越小，显存峰值越低。64MB 在 6GB 显卡上是一个较稳妥的折中。
_MAX_IM2COL_BYTES = 64 * 1024 * 1024


# ======================================================================
# 逐元素算子的反向传播用的是“乘上游梯度”的链式法则
# ======================================================================
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        # 两个输入都可能被广播，反向传播时需要把梯度规约回各自形状
        ctx["shapes"] = (a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx, grad):
        sa, sb = ctx["shapes"]
        return _unbroadcast(grad, sa), _unbroadcast(grad, sb)


class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx["shapes"] = (a.shape, b.shape)
        return a - b

    @staticmethod
    def backward(ctx, grad):
        sa, sb = ctx["shapes"]
        return _unbroadcast(grad, sa), _unbroadcast(-grad, sb)


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx["a"] = a
        ctx["b"] = b
        ctx["shapes"] = (a.shape, b.shape)
        return a * b

    @staticmethod
    def backward(ctx, grad):
        sa, sb = ctx["shapes"]
        return (
            _unbroadcast(grad * ctx["b"], sa),
            _unbroadcast(grad * ctx["a"], sb),
        )


class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx["a"] = a
        ctx["b"] = b
        ctx["shapes"] = (a.shape, b.shape)
        return a / b

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx["a"], ctx["b"]
        sa, sb = ctx["shapes"]
        return (
            _unbroadcast(grad / b, sa),
            _unbroadcast(-grad * a / (b * b), sb),
        )


class Neg(Function):
    @staticmethod
    def forward(ctx, a):
        return -a

    @staticmethod
    def backward(ctx, grad):
        return (-grad,)


class Pow(Function):
    """``x ** exponent``，exponent 为标量常量。"""

    @staticmethod
    def forward(ctx, x):
        ctx["x"] = x
        return x ** ctx["exponent"]

    @staticmethod
    def backward(ctx, grad):
        x = ctx["x"]
        e = ctx["exponent"]
        return (grad * e * x ** (e - 1),)


# ======================================================================
# 矩阵乘 / 形状操作 / 归约
# ======================================================================
class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx["a"] = a
        ctx["b"] = b
        ctx["shapes"] = (a.shape, b.shape)
        return cp.matmul(a, b)

    @staticmethod
    def backward(ctx, grad):
        a, b = ctx["a"], ctx["b"]
        sa, sb = ctx["shapes"]
        # 支持 (n, k) @ (k, m) 以及批量维广播的 (n, k) @ (b, k, m)
        da = cp.matmul(grad, cp.swapaxes(b, -1, -2))
        db = cp.matmul(cp.swapaxes(a, -1, -2), grad)
        return _unbroadcast(da, sa), _unbroadcast(db, sb)


class Reshape(Function):
    @staticmethod
    def forward(ctx, x):
        ctx["shape"] = x.shape
        return x.reshape(ctx["new_shape"])

    @staticmethod
    def backward(ctx, grad):
        return (grad.reshape(ctx["shape"]),)


class Transpose(Function):
    @staticmethod
    def forward(ctx, x):
        axes = ctx["axes"]
        if axes is None:
            axes = tuple(range(x.ndim - 1, -1, -1))
            ctx["axes"] = axes
        ctx["inv"] = tuple(int(i) for i in cp.argsort(cp.asarray(axes)))
        return x.transpose(axes)

    @staticmethod
    def backward(ctx, grad):
        return (grad.transpose(ctx["inv"]),)


class GetItem(Function):
    """下标/切片视图，支持 ``[None, :, None, None]`` 这类插轴写法。

    反向传播时把梯度「散射」回原形状的空零数组上；对于
    ``None`` 新增的轴，直接在同一个 key 上做 ``+=`` 即可由 CuPy
    自动完成广播求和。
    """

    @staticmethod
    def forward(ctx, x):
        ctx["shape"] = x.shape
        return x[ctx["key"]]

    @staticmethod
    def backward(ctx, grad):
        dx = cp.zeros(ctx["shape"], dtype=grad.dtype)
        dx[ctx["key"]] += grad
        return (dx,)


class Sum(Function):
    @staticmethod
    def forward(ctx, x):
        ctx["shape"] = x.shape
        ctx["axis"] = ctx.get("axis", None)
        ctx["keepdims"] = ctx.get("keepdims", False)
        return x.sum(axis=ctx["axis"], keepdims=ctx["keepdims"])

    @staticmethod
    def backward(ctx, grad):
        return (_expand_reduced(grad, ctx["shape"], ctx["axis"], ctx["keepdims"]),)


class Mean(Function):
    @staticmethod
    def forward(ctx, x):
        ctx["shape"] = x.shape
        axis = ctx.get("axis", None)
        keepdims = ctx.get("keepdims", False)
        ctx["axis"] = axis
        ctx["keepdims"] = keepdims

        out = x.mean(axis=axis, keepdims=keepdims)
        # 被归约的元素个数，反向传播时需要除以它
        if axis is None:
            count = x.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = 1
            for a in axes:
                count *= x.shape[a]
        ctx["count"] = float(count)
        return out

    @staticmethod
    def backward(ctx, grad):
        g = _expand_reduced(grad, ctx["shape"], ctx["axis"], ctx["keepdims"])
        return (g / ctx["count"],)


def _expand_reduced(grad, shape, axis, keepdims):
    """把归约后的梯度广播回原始形状。"""
    if axis is None:
        return cp.broadcast_to(grad, shape).copy()

    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    ndim = len(shape)
    axes = tuple(a % ndim for a in axes)
    if not keepdims:
        grad = cp.expand_dims(grad, axis=axes)
    return cp.broadcast_to(grad, shape).copy()


# ======================================================================
# 激活函数
# ======================================================================
class Relu(Function):
    @staticmethod
    def forward(ctx, x):
        ctx["mask"] = x > 0
        return cp.maximum(x, 0, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return (grad * ctx["mask"],)


class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        out = cp.exp(x)
        ctx["out"] = out
        return out

    @staticmethod
    def backward(ctx, grad):
        return (grad * ctx["out"],)


class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx["x"] = x
        return cp.log(x)

    @staticmethod
    def backward(ctx, grad):
        return (grad / ctx["x"],)


# ======================================================================
# 二维卷积（手写 im2col + 分块矩阵乘）
# ======================================================================
def _im2col(xp: cp.ndarray, kh: int, kw: int, stride: int, oh: int, ow: int):
    """把填充后的特征图展开成 (N, C*kh*kw, oh*ow) 的列矩阵。

    ``xp`` 形状为 (N, C, H+2p, W+2p)。逐 (i, j) 取出滑动窗口，
    写入 col 的对应通道块，避免 as_strided 造成的不可控拷贝。
    """
    n, c = xp.shape[0], xp.shape[1]
    # 先按 (N, C, kh, kw, oh, ow) 布局写入，最后 reshape 成矩阵形式，
    # 这样 w.reshape(O, -1) 与 col 的 K 维顺序严格一致。
    col = cp.empty((n, c, kh, kw, oh, ow), dtype=xp.dtype)
    for i in range(kh):
        for j in range(kw):
            col[:, :, i, j] = xp[
                :, :, i : i + stride * oh : stride, j : j + stride * ow : stride
            ]
    return col.reshape(n, c * kh * kw, oh * ow)


def _col2im(dcol: cp.ndarray, n: int, c: int, kh: int, kw: int,
            oh: int, ow: int, h: int, w: int, stride: int, dtype):
    """im2col 的逆操作：把列梯度累加回输入特征图。

    返回形状为 (N, C, (oh-1)*stride+kh, (ow-1)*stride+kw) 的梯度，
    调用方再裁掉 padding 区域。
    """
    dcol = dcol.reshape(n, c, kh, kw, oh, ow)
    hpad = (oh - 1) * stride + kh
    wpad = (ow - 1) * stride + kw
    dxp = cp.zeros((n, c, hpad, wpad), dtype=dtype)
    # 不同 (i, j) 对应的位置互不重叠，但都需要累加，因此用 +=
    for i in range(kh):
        for j in range(kw):
            dxp[:, :, i : i + stride * oh : stride, j : j + stride * ow : stride] += dcol[
                :, :, i, j
            ]
    return dxp


class Conv2dFn(Function):
    @staticmethod
    def forward(ctx, x, weight):
        n, c, h, w = x.shape
        out_channels, _, kh, kw = weight.shape
        stride = ctx["stride"]
        padding = ctx["padding"]

        oh = (h + 2 * padding - kh) // stride + 1
        ow = (w + 2 * padding - kw) // stride + 1

        if padding > 0:
            xp = cp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        else:
            xp = x

        # 先按 (N, O, oh*ow) 分配，矩阵乘结果可直接写入，最后再 reshape
        out = cp.empty((n, out_channels, oh * ow), dtype=x.dtype)
        w2 = weight.reshape(out_channels, -1)

        # 每块 im2col 的字节数 = chunk * C*kh*kw*oh*ow*itemsize
        per_sample = c * kh * kw * oh * ow * x.dtype.itemsize
        chunk = max(1, min(n, _MAX_IM2COL_BYTES // max(per_sample, 1)))

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            col = _im2col(xp[s:e], kh, kw, stride, oh, ow)
            out[s:e] = cp.matmul(w2, col)
        out = out.reshape(n, out_channels, oh, ow)

        # 保存反向传播所需信息（只保存引用/形状，不额外拷贝大数组）
        ctx["x"] = x
        ctx["w"] = weight
        ctx["oh"], ctx["ow"] = oh, ow
        ctx["h"], ctx["w_in"] = h, w
        ctx["n"], ctx["c"], ctx["kh"], ctx["kw"] = n, c, kh, kw
        return out

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx["x"], ctx["w"]
        stride, padding = ctx["stride"], ctx["padding"]
        n, c = ctx["n"], ctx["c"]
        kh, kw = ctx["kh"], ctx["kw"]
        oh, ow = ctx["oh"], ctx["ow"]
        h, w = ctx["h"], ctx["w_in"]

        if padding > 0:
            xp = cp.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
        else:
            xp = x

        out_channels = weight.shape[0]
        w2 = weight.reshape(out_channels, -1)
        dw = cp.zeros((out_channels, c * kh * kw), dtype=weight.dtype)

        per_sample = c * kh * kw * oh * ow * x.dtype.itemsize
        chunk = max(1, min(n, _MAX_IM2COL_BYTES // max(per_sample, 1)))

        # 输入梯度缓冲区（含 padding）。所有分块都累加到同一块缓冲区上。
        dxp_full = cp.zeros_like(xp)

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            col = _im2col(xp[s:e], kh, kw, stride, oh, ow)
            g = grad[s:e].reshape(e - s, out_channels, -1)
            # dW = sum_n (dY_n @ col_n^T)
            dw += cp.matmul(g, col.transpose(0, 2, 1)).sum(axis=0)
            # dcol = W^T @ dY，再散射回输入特征图
            dcol = cp.matmul(w2.T, g)
            dxp = _col2im(dcol, e - s, c, kh, kw, oh, ow, h, w, stride, x.dtype)
            # _col2im 的尺寸是 (oh-1)*stride+kh，在 stride>1 时可能比「输入+padding」
            # 略小（右下角那部分 padding 根本没被窗口访问到），因此对齐到左上角累加。
            dxp_full[s:e, :, : dxp.shape[2], : dxp.shape[3]] += dxp

        # 裁掉 padding 区域，得到与输入完全一致的形状
        dx = dxp_full[:, :, padding : padding + h, padding : padding + w]
        return dx, dw.reshape(weight.shape)


def conv2d(x: Tensor, weight: Tensor, stride: int = 1, padding: int = 0) -> Tensor:
    """2D 卷积（PyTorch 风格 NCHW 布局）。"""
    return Conv2dFn.apply(x, weight, stride=int(stride), padding=int(padding))


# ======================================================================
# BatchNorm2d
# ======================================================================
class BatchNorm2dFn(Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        n, c, h, w = x.shape
        eps = ctx["eps"]
        momentum = ctx["momentum"]
        training = ctx["training"]
        running_mean = ctx["running_mean"]
        running_var = ctx["running_var"]

        m = n * h * w  # 每个通道参与统计的元素个数

        if training:
            mean = x.mean(axis=(0, 2, 3))
            # 归一化使用有偏方差；running_var 使用无偏方差（与 PyTorch 一致）
            var = x.var(axis=(0, 2, 3))
            running_mean *= 1.0 - momentum
            running_mean += momentum * mean
            running_var *= 1.0 - momentum
            running_var += momentum * (var * (m / max(m - 1, 1)))
        else:
            mean = running_mean
            var = running_var

        inv_std = 1.0 / cp.sqrt(var + eps)
        xhat = (x - mean[None, :, None, None]) * inv_std[None, :, None, None]
        out = xhat * weight[None, :, None, None] + bias[None, :, None, None]

        # 保存反向传播所需信息
        ctx["xhat"] = xhat
        ctx["inv_std"] = inv_std
        ctx["m"] = m
        # forward 收到的 weight 已经是 cupy 数组，这里保存引用供 backward 使用
        ctx["weight_tensor"] = weight
        return out

    @staticmethod
    def backward(ctx, grad):
        xhat = ctx["xhat"]
        inv_std = ctx["inv_std"]
        m = ctx["m"]
        weight = ctx["weight_tensor"]
        training = ctx["training"]

        dgamma = (grad * xhat).sum(axis=(0, 2, 3))
        dbeta = grad.sum(axis=(0, 2, 3))

        dxhat = grad * weight[None, :, None, None]
        if training:
            # 标准 BatchNorm 反向公式：
            # dx = inv_std * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
            sum_dxhat = dxhat.sum(axis=(0, 2, 3))
            sum_dxhat_xhat = (dxhat * xhat).sum(axis=(0, 2, 3))
            dx = inv_std[None, :, None, None] * (
                dxhat
                - sum_dxhat[None, :, None, None] / m
                - xhat * (sum_dxhat_xhat[None, :, None, None] / m)
            )
        else:
            dx = dxhat * inv_std[None, :, None, None]

        return dx, dgamma, dbeta


def batch_norm(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    running_mean,
    running_var,
    eps: float = 1e-5,
    momentum: float = 0.1,
    training: bool = True,
) -> Tensor:
    """2D BatchNorm。

    running_mean / running_var 是普通的 cupy 数组（非 Tensor），
    在 forward 中以原地方式更新，因此不会进入计算图。
    """
    return BatchNorm2dFn.apply(
        x,
        weight,
        bias,
        eps=eps,
        momentum=momentum,
        training=training,
        running_mean=running_mean,
        running_var=running_var,
    )


# ======================================================================
# MaxPool2d
# ======================================================================
class MaxPool2dFn(Function):
    @staticmethod
    def forward(ctx, x):
        n, c, h, w = x.shape
        k = ctx["kernel_size"]
        s = ctx["stride"]

        assert h % k == 0 and w % k == 0, "MaxPool 要求输入尺寸能被池化核整除"
        oh, ow = h // k, w // k

        # (N, C, H, W) -> (N, C, oh, k, ow, k) -> (N, C, oh, ow, k*k)
        v = x.reshape(n, c, oh, k, ow, k).transpose(0, 1, 2, 4, 3, 5)
        v = cp.ascontiguousarray(v).reshape(n, c, oh, ow, k * k)
        out = v.max(axis=-1)

        # 记录最大值位置；平局时把梯度平均分给所有最大值，避免梯度放大
        mask = v == out[..., None]
        count = mask.sum(axis=-1, keepdims=True)

        ctx["shape"] = (n, c, h, w)
        ctx["oh"], ctx["ow"], ctx["k"] = oh, ow, k
        ctx["mask"] = mask
        ctx["count"] = count
        return out

    @staticmethod
    def backward(ctx, grad):
        n, c, h, w = ctx["shape"]
        oh, ow, k = ctx["oh"], ctx["ow"], ctx["k"]
        mask, count = ctx["mask"], ctx["count"]

        g = grad[..., None] * mask / count
        g = g.reshape(n, c, oh, ow, k, k)
        # 逆变换 (N, C, oh, ow, k, k) -> (N, C, oh, k, ow, k) -> (N, C, H, W)
        g = g.transpose(0, 1, 2, 4, 3, 5)
        return (cp.ascontiguousarray(g).reshape(n, c, h, w),)


def max_pool2d(x: Tensor, kernel_size: int = 2, stride: int | None = None) -> Tensor:
    if stride is None:
        stride = kernel_size
    if stride != kernel_size:
        raise NotImplementedError("当前 MaxPool 只支持 stride == kernel_size")
    return MaxPool2dFn.apply(x, kernel_size=int(kernel_size), stride=int(stride))


# ======================================================================
# 交叉熵（内部融合 log-softmax，数值稳定）
# ======================================================================
class CrossEntropyFn(Function):
    @staticmethod
    def forward(ctx, logits):
        targets = ctx["targets"]
        smoothing = ctx["label_smoothing"]
        n, c = logits.shape

        # 减去每行最大值，避免 exp 溢出
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_sum_exp = cp.log(cp.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        rows = cp.arange(n)
        nll = -log_probs[rows, targets]

        if smoothing > 0:
            smooth = -log_probs.sum(axis=1) / c
            loss = ((1.0 - smoothing) * nll + smoothing * smooth).mean()
        else:
            loss = nll.mean()

        ctx["log_probs"] = log_probs
        ctx["n"] = n
        return loss

    @staticmethod
    def backward(ctx, grad):
        log_probs = ctx["log_probs"]
        targets = ctx["targets"]
        smoothing = ctx["label_smoothing"]
        n, c = log_probs.shape
        rows = cp.arange(n)

        # dL/dlogits = softmax - (1-eps)*onehot - eps/C
        d = cp.exp(log_probs)
        if smoothing > 0:
            d[rows, targets] -= 1.0 - smoothing
            d -= smoothing / c
        else:
            d[rows, targets] -= 1.0

        d *= grad / n
        return (d,)


def cross_entropy(
    logits: Tensor, targets, label_smoothing: float = 0.0
) -> Tensor:
    """softmax 交叉熵。``targets`` 可以是 cupy 数组或 CPU 数组。"""
    if isinstance(targets, Tensor):
        targets = targets.data
    targets = cp.asarray(targets, dtype=cp.int32)
    return CrossEntropyFn.apply(
        logits, targets=targets, label_smoothing=float(label_smoothing)
    )


# ======================================================================
# 便捷包装函数
# ======================================================================
def add(a, b) -> Tensor:
    return Add.apply(a, b)


def sub(a, b) -> Tensor:
    return Sub.apply(a, b)


def mul(a, b) -> Tensor:
    return Mul.apply(a, b)


def div(a, b) -> Tensor:
    return Div.apply(a, b)


def neg(a) -> Tensor:
    return Neg.apply(a)


def pow(a, exponent: float) -> Tensor:
    return Pow.apply(a, exponent=exponent)


def matmul(a, b) -> Tensor:
    return MatMul.apply(a, b)


def reshape(x, shape) -> Tensor:
    return Reshape.apply(x, new_shape=tuple(shape))


def transpose(x, axes=None) -> Tensor:
    return Transpose.apply(x, axes=axes)


def getitem(x, key) -> Tensor:
    return GetItem.apply(x, key=key)


def sum(x, axis=None, keepdims: bool = False) -> Tensor:
    return Sum.apply(x, axis=axis, keepdims=keepdims)


def mean(x, axis=None, keepdims: bool = False) -> Tensor:
    return Mean.apply(x, axis=axis, keepdims=keepdims)


def relu(x) -> Tensor:
    return Relu.apply(x)


def exp(x) -> Tensor:
    return Exp.apply(x)


def log(x) -> Tensor:
    return Log.apply(x)
