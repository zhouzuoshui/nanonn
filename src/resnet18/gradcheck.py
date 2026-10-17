"""手写算子的数值梯度自检（gradient check）。

思路：用 float64 的 numpy 数组作为「真值容器」，对每个元素做中心差分

    ∂L/∂x_i ≈ (L(x_i + ε) - L(x_i - ε)) / (2ε)

得到数值梯度，再和手写的 ``backward`` 结果比较相对误差。

为什么需要它：手写反向传播最容易在「分块」「stride/padding 组合」「广播规约」
这些地方写错，而这类错误在训练时只表现为「loss 不降」，很难定位。

用法::

    uv run python -m resnet18.gradcheck

全部通过时退出码为 0，否则为 1（可以直接接进 CI）。
"""

from __future__ import annotations

import sys

import cupy as cp
import numpy as np

from nanonn import functional as F
from nanonn.tensor import Tensor

# 相对误差阈值：float64 + eps=1e-5 时，正确实现的误差通常在 1e-9 量级
DEFAULT_TOL = 1e-6

# 记录失败的用例名，最后统一汇报
FAILED: list[str] = []


def _to_cp(arrays):
    """numpy float64 数组 -> cupy float64 数组列表。"""
    return [cp.asarray(a, dtype=cp.float64) for a in arrays]


def _num_grad(loss_fn, arrays, eps: float = 1e-5):
    """对 ``arrays`` 里每个数组逐元素做中心差分。

    ``loss_fn`` 接受「numpy 数组列表」并返回 float 的 loss 值。
    这里直接操作 ``reshape(-1)`` 视图，避免使用 ``np.nditer``
    （对非连续数组会触发 WRITEBACKIFCOPY 错误）。
    """
    grads = []
    for arr in arrays:
        g = np.zeros_like(arr)
        flat = arr.reshape(-1)
        gflat = g.reshape(-1)
        for k in range(flat.size):
            old = flat[k]
            flat[k] = old + eps
            fp = loss_fn(arrays)
            flat[k] = old - eps
            fm = loss_fn(arrays)
            flat[k] = old
            gflat[k] = (fp - fm) / (2.0 * eps)
        grads.append(g)
    return grads


def _check(name: str, analytic, numeric, tol: float) -> None:
    a = cp.asnumpy(analytic).astype(np.float64)
    err = np.max(np.abs(a - numeric)) / (np.max(np.abs(numeric)) + 1e-12)
    ok = err < tol
    print(f"{'OK  ' if ok else 'FAIL'} {name:<22} rel_err={err:.3e}")
    if not ok:
        FAILED.append(name)


def _run_case(name, fwd_backward, loss_fn, arrays, tol: float = DEFAULT_TOL) -> None:
    """跑一个用例：先解析求梯度，再数值求梯度，逐输入比较。"""
    analytic = fwd_backward()
    numeric = _num_grad(loss_fn, arrays)
    for i, (a, n) in enumerate(zip(analytic, numeric)):
        _check(f"{name}[{i}]", a, n, tol)


# ======================================================================
# conv2d：覆盖 stride/padding 的各种组合、以及分块路径
# ======================================================================
def case_conv() -> None:
    rng = np.random.default_rng(0)

    # --- stride=1, padding=1（ResNet 里最常见的 3x3 same 卷积）---
    x = rng.standard_normal((2, 3, 6, 6))
    w = rng.standard_normal((4, 3, 3, 3))
    R = rng.standard_normal((2, 4, 6, 6))  # 上游梯度（随机权重让每个元素都参与）

    def fb():
        xt = Tensor(cp.asarray(x), True)
        wt = Tensor(cp.asarray(w), True)
        out = F.conv2d(xt, wt, stride=1, padding=1)
        (out * Tensor(cp.asarray(R), False)).sum().backward()
        return [xt.grad, wt.grad]

    def loss(arrs):
        xc, wc = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.conv2d(Tensor(xc), Tensor(wc), stride=1, padding=1)
                 * Tensor(cp.asarray(R), False)).sum().data
            )
        )

    _run_case("conv_s1p1", fb, loss, [x, w])

    # --- stride=2, padding=1（下采样卷积）---
    x2 = rng.standard_normal((2, 3, 7, 7))
    w2 = rng.standard_normal((3, 3, 3, 3))
    R2 = rng.standard_normal((2, 3, 4, 4))

    def fb2():
        xt = Tensor(cp.asarray(x2), True)
        wt = Tensor(cp.asarray(w2), True)
        out = F.conv2d(xt, wt, stride=2, padding=1)
        assert out.shape == (2, 3, 4, 4), out.shape
        (out * Tensor(cp.asarray(R2), False)).sum().backward()
        return [xt.grad, wt.grad]

    def loss2(arrs):
        xc, wc = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.conv2d(Tensor(xc), Tensor(wc), stride=2, padding=1)
                 * Tensor(cp.asarray(R2), False)).sum().data
            )
        )

    _run_case("conv_s2p1", fb2, loss2, [x2, w2])

    # --- stride=2 且 (oh-1)*stride+kh < h+2*padding ---
    # 例如输入 8x8、padding=1：右边/下边那一圈 padding 根本没被窗口访问到，
    # col2im 的结果会比「输入+padding」小 1，这里专门验证这种情况不会出错。
    x2b = rng.standard_normal((1, 2, 8, 8))
    w2b = rng.standard_normal((3, 2, 3, 3))
    R2b = rng.standard_normal((1, 3, 4, 4))

    def fb2b():
        xt = Tensor(cp.asarray(x2b), True)
        wt = Tensor(cp.asarray(w2b), True)
        out = F.conv2d(xt, wt, stride=2, padding=1)
        assert out.shape == (1, 3, 4, 4), out.shape
        (out * Tensor(cp.asarray(R2b), False)).sum().backward()
        return [xt.grad, wt.grad]

    def loss2b(arrs):
        xc, wc = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.conv2d(Tensor(xc), Tensor(wc), stride=2, padding=1)
                 * Tensor(cp.asarray(R2b), False)).sum().data
            )
        )

    _run_case("conv_s2p1_ragged", fb2b, loss2b, [x2b, w2b])

    # --- 强制走分块路径（把 im2col 预算压到 1 字节，chunk 退化为 1）---
    # ResNet 训练时 batch=128 会真正分块，这个用例保证「多块累加」是对的。
    x2c = rng.standard_normal((3, 2, 6, 6))
    w2c = rng.standard_normal((3, 2, 3, 3))
    R2c = rng.standard_normal((3, 3, 3, 3))

    def fb2c():
        old = F._MAX_IM2COL_BYTES
        F._MAX_IM2COL_BYTES = 1
        try:
            xt = Tensor(cp.asarray(x2c), True)
            wt = Tensor(cp.asarray(w2c), True)
            out = F.conv2d(xt, wt, stride=2, padding=1)
            assert out.shape == (3, 3, 3, 3), out.shape
            (out * Tensor(cp.asarray(R2c), False)).sum().backward()
            return [xt.grad, wt.grad]
        finally:
            F._MAX_IM2COL_BYTES = old

    def loss2c(arrs):
        xc, wc = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.conv2d(Tensor(xc), Tensor(wc), stride=2, padding=1)
                 * Tensor(cp.asarray(R2c), False)).sum().data
            )
        )

    _run_case("conv_chunked", fb2c, loss2c, [x2c, w2c])

    # --- 无 padding（1x1 卷积之外的另一条分支）---
    x3 = rng.standard_normal((1, 2, 5, 5))
    w3 = rng.standard_normal((3, 2, 3, 3))
    R3 = rng.standard_normal((1, 3, 3, 3))

    def fb3():
        xt = Tensor(cp.asarray(x3), True)
        wt = Tensor(cp.asarray(w3), True)
        out = F.conv2d(xt, wt, stride=1, padding=0)
        (out * Tensor(cp.asarray(R3), False)).sum().backward()
        return [xt.grad, wt.grad]

    def loss3(arrs):
        xc, wc = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.conv2d(Tensor(xc), Tensor(wc), stride=1, padding=0)
                 * Tensor(cp.asarray(R3), False)).sum().data
            )
        )

    _run_case("conv_s1p0", fb3, loss3, [x3, w3])


# ======================================================================
# BatchNorm2d（训练模式：用 batch 统计量）
# ======================================================================
def case_bn() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal((3, 4, 4, 4))
    gamma = rng.standard_normal(4)
    beta = rng.standard_normal(4)
    R = rng.standard_normal((3, 4, 4, 4))

    def fb():
        xt = Tensor(cp.asarray(x), True)
        gt = Tensor(cp.asarray(gamma), True)
        bt = Tensor(cp.asarray(beta), True)
        out = F.batch_norm(
            xt, gt, bt, cp.zeros(4), cp.ones(4),
            eps=1e-5, momentum=0.1, training=True,
        )
        (out * Tensor(cp.asarray(R), False)).sum().backward()
        return [xt.grad, gt.grad, bt.grad]

    def loss(arrs):
        xc, gc, bc = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.batch_norm(Tensor(xc), Tensor(gc), Tensor(bc),
                              cp.zeros(4), cp.ones(4),
                              eps=1e-5, momentum=0.1, training=True)
                 * Tensor(cp.asarray(R), False)).sum().data
            )
        )

    _run_case("batch_norm", fb, loss, [x, gamma, beta], tol=1e-5)


# ======================================================================
# MaxPool2d
# ======================================================================
def case_pool() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 4, 4))
    R = rng.standard_normal((2, 3, 2, 2))

    def fb():
        xt = Tensor(cp.asarray(x), True)
        out = F.max_pool2d(xt, 2, 2)
        (out * Tensor(cp.asarray(R), False)).sum().backward()
        return [xt.grad]

    def loss(arrs):
        (xc,) = _to_cp(arrs)
        return float(
            cp.asnumpy(
                (F.max_pool2d(Tensor(xc), 2, 2)
                 * Tensor(cp.asarray(R), False)).sum().data
            )
        )

    _run_case("max_pool2d", fb, loss, [x])


# ======================================================================
# CrossEntropy（含 label smoothing 分支）
# ======================================================================
def case_ce() -> None:
    rng = np.random.default_rng(3)
    logits = rng.standard_normal((5, 7))
    targets = np.array([0, 3, 6, 2, 5], dtype=np.int32)

    for ls in (0.0, 0.1):
        def fb(ls=ls):
            lt = Tensor(cp.asarray(logits), True)
            F.cross_entropy(lt, targets, label_smoothing=ls).backward()
            return [lt.grad]

        def loss(arrs, ls=ls):
            (lc,) = _to_cp(arrs)
            return float(
                cp.asnumpy(
                    F.cross_entropy(Tensor(lc), targets, label_smoothing=ls).data
                )
            )

        _run_case(f"cross_entropy(ls={ls})", fb, loss, [logits], tol=1e-5)


# ======================================================================
# 通用算子：matmul / transpose / relu / mean / 广播加法（对应 fc 与 bias）
# ======================================================================
def case_misc() -> None:
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 4))
    W = rng.standard_normal((5, 4))

    def fb():
        xt = Tensor(cp.asarray(x), True)
        wt = Tensor(cp.asarray(W), True)
        out = F.matmul(xt, F.transpose(wt, (1, 0)))  # (3,5)
        out = F.relu(out)
        out = F.mean(out * out, axis=1)              # (3,)
        F.sum(out).backward()
        return [xt.grad, wt.grad]

    def loss(arrs):
        xc, Wc = _to_cp(arrs)
        out = F.matmul(Tensor(xc), F.transpose(Tensor(Wc), (1, 0)))
        out = F.relu(out)
        return float(cp.asnumpy(F.sum(F.mean(out * out, axis=1)).data))

    _run_case("linear+relu", fb, loss, [x, W], tol=1e-5)

    # 广播：bias 用 [None, :] 插轴后相加（Conv2d/Linear 的实现方式）
    b = rng.standard_normal(5)
    Rb = rng.standard_normal((3, 5))

    def fb_bias():
        xt = Tensor(cp.asarray(x), True)
        bt = Tensor(cp.asarray(b), True)
        out = F.matmul(xt, F.transpose(Tensor(cp.asarray(W)), (1, 0)))
        out = out + bt[None, :]
        (out * Tensor(cp.asarray(Rb), False)).sum().backward()
        return [xt.grad, bt.grad]

    def loss_bias(arrs):
        xc, bc = _to_cp(arrs)
        out = F.matmul(Tensor(xc), F.transpose(Tensor(cp.asarray(W)), (1, 0)))
        out = out + Tensor(bc)[None, :]
        return float(cp.asnumpy((out * Tensor(cp.asarray(Rb), False)).sum().data))

    _run_case("matmul+bias", fb_bias, loss_bias, [x, b], tol=1e-5)


def main() -> int:
    case_conv()
    case_bn()
    case_pool()
    case_ce()
    case_misc()
    print()
    if FAILED:
        print(f"FAILED: {FAILED}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
