"""训练/推理共用的小工具：日志、计时、精度统计、权重存取。

权重保存为 ``.npz``（NumPy 格式），键名带 ``model.`` 前缀，
因此同一个文件里还能顺带存 epoch、最佳精度等元信息，
加载时只用 ``model.`` 前缀的项去填充网络。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from nanonn import functional as F
from nanonn.nn import Module
from nanonn.tensor import no_grad

__all__ = [
    "AverageMeter",
    "accuracy",
    "evaluate",
    "gpu_memory",
    "human_bytes",
    "load_checkpoint",
    "save_checkpoint",
    "set_seed",
    "format_seconds",
]


def set_seed(seed: int) -> None:
    """固定 CuPy（以及其背后的 cuRAND）随机种子，保证可复现。"""
    cp.random.seed(seed)


class AverageMeter:
    """按样本数加权的滑动平均，用于统计 loss / accuracy。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def accuracy(logits: cp.ndarray, targets: cp.ndarray) -> int:
    """返回一个 batch 中预测正确的样本数（top-1）。"""
    pred = logits.argmax(axis=1)
    return int((pred == targets).sum())


def evaluate(
    model: Module,
    dataset,
    batch_size: int = 256,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    """在测试集上计算平均交叉熵与 top-1 精度，返回 ``(loss, acc)``。

    评估阶段使用 ``no_grad``：不构建计算图，显存占用仅为前向激活。
    """
    model.eval()
    loss_meter = AverageMeter()
    correct = 0
    total = 0

    with no_grad():
        for x, y in dataset.test_batches(batch_size):
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
            n = int(y.shape[0])
            loss_meter.update(float(loss.data), n)
            correct += accuracy(logits.data, y)
            total += n
            # 及时释放，避免图/中间张量在下一轮迭代前仍被引用
            del logits, loss, x

    model.train()
    return loss_meter.avg, correct / max(total, 1)


def gpu_memory() -> tuple[int, int]:
    """返回当前设备 ``(已用字节, 总字节)``。"""
    free, total = cp.cuda.Device(0).mem_info
    return total - free, total


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f}{unit}"
        num /= 1024.0
    return f"{num:.1f}TB"


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def save_checkpoint(
    path: str | Path,
    model: Module,
    epoch: int | None = None,
    best_acc: float | None = None,
    extra: dict | None = None,
) -> None:
    """把模型参数（含 BN 的 running stats）保存为压缩 npz。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for name, value in model.state_dict().items():
        # cupy -> numpy；float16 等保持不变，加载时再对齐 dtype
        arrays[f"model.{name}"] = cp.asnumpy(value)
    arrays["meta.epoch"] = np.asarray(-1 if epoch is None else epoch, dtype=np.int64)
    arrays["meta.best_acc"] = np.asarray(
        float("nan") if best_acc is None else best_acc, dtype=np.float32
    )
    arrays["meta.extra"] = np.asarray(json.dumps(extra or {}))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path: str | Path, model: Module) -> tuple[int, float, dict]:
    """加载权重到 ``model``（原地拷贝），返回 ``(epoch, best_acc, extra)``。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到权重文件 {path}")

    with np.load(path, allow_pickle=False) as data:
        state = {}
        for key in data.files:
            if not key.startswith("model."):
                continue
            state[key[len("model."):]] = data[key]
        model.load_state_dict(state, strict=True)

        epoch = int(data["meta.epoch"]) if "meta.epoch" in data.files else -1
        best = float(data["meta.best_acc"]) if "meta.best_acc" in data.files else float("nan")
        extra = {}
        if "meta.extra" in data.files:
            try:
                extra = json.loads(str(data["meta.extra"].item()))
            except (json.JSONDecodeError, ValueError):  # pragma: no cover
                extra = {}

    return epoch, best, extra


class Timer:
    """简单的秒表，训练日志里用来统计每 epoch 耗时。"""

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def reset(self) -> None:
        self.start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start
