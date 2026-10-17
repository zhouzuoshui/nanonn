"""CIFAR-100 数据加载与 GPU 端数据增广（纯 CuPy / NumPy 实现）。

数据文件位于 ``data/cifar100/raw/cifar-100-python.tar.gz``，内部是 Python2
序列化的 pickle 字典：

- ``data``       : (N, 3072) uint8，前 1024 字节是 R，随后 G，最后 B（按行展开）
- ``fine_labels``: (N,) int，0-99
- ``meta``       : 含 ``fine_label_names``（100 个类别名）

为了少占内存、也为了不让 Python 的 pickle 解析成为瓶颈，这里：
1. 一次性从 tar 包直接读出训练集/测试集并整理成 (N, 3, 32, 32) uint8；
2. 全部搬到显存（约 2 x 150MB，3060 6GB 显存完全够用）；
3. 训练时在 GPU 上做随机裁剪 + 随机水平翻转（向量化，无需 DataLoader 线程）。
"""

from __future__ import annotations

import pickle
import tarfile
from pathlib import Path
from typing import Iterator

import cupy as cp
import numpy as np

from nanonn.tensor import Tensor

__all__ = ["CIFAR100", "CIFAR100_MEAN", "CIFAR100_STD", "load_cifar100"]

# 项目根目录下的数据目录（src/resnet18/data.py -> 上两级）
DEFAULT_CIFAR100_ROOT = Path(__file__).resolve().parents[2] / "data" / "cifar100"

# CIFAR-100 训练集的逐通道均值/标准差（0-1 归一化前提下）
CIFAR100_MEAN = (0.50707516, 0.48654887, 0.44091784)
CIFAR100_STD = (0.26733429, 0.25643846, 0.27615047)


def _get_field(record: dict, key: str):
    """pickle 的键可能是 str 也可能是 bytes，这里统一处理。"""
    if key in record:
        return record[key]
    if key.encode() in record:
        return record[key.encode()]
    raise KeyError(f"CIFAR-100 文件中缺少字段 {key!r}")


def _load_pickle_member(tar: tarfile.TarFile, name: str) -> dict:
    member = tar.extractfile(name)
    if member is None:  # pragma: no cover - 文件损坏时才会发生
        raise FileNotFoundError(f"tar 包中找不到 {name}")
    # encoding="latin1" 用于兼容 Python2 序列化的数据
    return pickle.loads(member.read(), encoding="latin1")


def load_cifar100(root: str | Path | None = None):
    """从 tar.gz（或已解压目录）读取 CIFAR-100。

    返回 ``(train_x, train_y, test_x, test_y, label_names, train_filenames,
    test_filenames)``，其中图像形状为 (N, 3, 32, 32) 的 uint8 数组。
    """
    root = Path(root) if root is not None else DEFAULT_CIFAR100_ROOT
    if not root.exists():
        raise FileNotFoundError(
            f"找不到数据目录 {root}，请把 CIFAR-100 放到 data/cifar100/raw 下"
        )

    # 依次尝试几种常见位置
    candidates = [
        root / "raw" / "cifar-100-python.tar.gz",
        root / "cifar-100-python.tar.gz",
        root / "raw" / "cifar-100-python",
        root / "cifar-100-python",
    ]
    source = next((p for p in candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            f"在 {root} 下没找到 cifar-100-python.tar.gz 或 cifar-100-python 目录"
        )

    if source.is_dir():
        def read(name: str) -> dict:
            with open(source / name, "rb") as f:
                return pickle.load(f, encoding="latin1")

        meta = read("meta")
        train = read("train")
        test = read("test")
    else:
        with tarfile.open(source, "r:gz") as tar:
            meta = _load_pickle_member(tar, "cifar-100-python/meta")
            train = _load_pickle_member(tar, "cifar-100-python/train")
            test = _load_pickle_member(tar, "cifar-100-python/test")

    def unpack(record: dict):
        # (N, 3072) -> (N, 3, 32, 32)
        x = np.asarray(_get_field(record, "data"), dtype=np.uint8)
        x = x.reshape(-1, 3, 32, 32)
        y = np.asarray(_get_field(record, "fine_labels"), dtype=np.int64)
        names = list(_get_field(record, "filenames"))
        return x, y, names

    train_x, train_y, train_names = unpack(train)
    test_x, test_y, test_names = unpack(test)
    label_names = [n.decode() if isinstance(n, bytes) else str(n)
                   for n in _get_field(meta, "fine_label_names")]
    return train_x, train_y, test_x, test_y, label_names, train_names, test_names


class CIFAR100:
    """显存常驻的 CIFAR-100 数据集，提供按 batch 迭代的接口。"""

    def __init__(self, root: str | Path | None = None):
        train_x, train_y, test_x, test_y, label_names, tr_names, te_names = (
            load_cifar100(root)
        )

        # 图像常驻显存：uint8 下 (50000,3,32,32) = 150MB，(10000,...) = 30MB
        self.train_images = cp.asarray(train_x)
        self.train_labels = cp.asarray(train_y)
        self.test_images = cp.asarray(test_x)
        self.test_labels = cp.asarray(test_y)

        self.label_names = label_names
        self.train_filenames = tr_names
        self.test_filenames = te_names
        self.num_classes = len(label_names)

        # 归一化参数，形状 (1,3,1,1) 便于广播
        self.mean = cp.asarray(CIFAR100_MEAN, dtype=cp.float32).reshape(1, 3, 1, 1)
        self.std = cp.asarray(CIFAR100_STD, dtype=cp.float32).reshape(1, 3, 1, 1)

    # ------------------------------------------------------------------
    # 数据增广
    # ------------------------------------------------------------------
    def _augment(self, images: cp.ndarray) -> cp.ndarray:
        """随机裁剪（4 像素 padding）+ 随机水平翻转，完全向量化。

        输入/输出都是 (N, 3, 32, 32) 的 uint8 数组。
        """
        n = images.shape[0]
        # 先四周补 0（CIFAR 图像本身做过零均值，边界补 0 等价于补黑）
        padded = cp.pad(images, ((0, 0), (0, 0), (4, 4), (4, 4)), mode="constant")

        # 每张图独立采样裁剪左上角偏移量，范围 0..8
        offsets_y = cp.random.randint(0, 9, size=n)
        offsets_x = cp.random.randint(0, 9, size=n)
        batch_idx = cp.arange(n)

        rows = offsets_y[:, None, None] + cp.arange(32)[None, :, None]  # (n,32,1)
        cols = offsets_x[:, None, None] + cp.arange(32)[None, None, :]  # (n,1,32)

        # 花式索引一次取出所有裁剪块；高级索引结果维度在前，需转置回 NCHW
        cropped = padded[batch_idx[:, None, None], :, rows, cols].transpose(0, 3, 1, 2)

        # 随机水平翻转：对整个 batch 用 where 处理，避免 python 循环
        flip = cp.random.random(n) < 0.5
        cropped = cp.where(flip[:, None, None, None], cropped[:, :, :, ::-1], cropped)
        # 花式索引产生的是新数组，continus 化以便后续 astype 高效
        return cp.ascontiguousarray(cropped)

    def _to_float(self, images: cp.ndarray) -> cp.ndarray:
        """uint8 -> float32 并做标准化（返回新数组，不修改数据集本身）。"""
        x = images.astype(cp.float32)
        x *= 1.0 / 255.0
        x -= self.mean
        x /= self.std
        return x

    # ------------------------------------------------------------------
    # batch 迭代
    # ------------------------------------------------------------------
    def _batch(self, images, labels, index, augment: bool) -> tuple[Tensor, cp.ndarray]:
        imgs = images[index]
        if augment:
            imgs = self._augment(imgs)
        x = self._to_float(imgs)
        return Tensor(x, requires_grad=False), labels[index]

    def train_batches(
        self,
        batch_size: int = 128,
        augment: bool = True,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[tuple[Tensor, cp.ndarray]]:
        """迭代训练集。默认打乱顺序并做数据增广。"""
        n = int(self.train_images.shape[0])
        order = cp.random.permutation(n) if shuffle else cp.arange(n)
        n_batches = n // batch_size if drop_last else (n + batch_size - 1) // batch_size

        for i in range(n_batches):
            index = order[i * batch_size:(i + 1) * batch_size]
            yield self._batch(self.train_images, self.train_labels, index, augment)

    def test_batches(
        self, batch_size: int = 256
    ) -> Iterator[tuple[Tensor, cp.ndarray]]:
        """迭代测试集：不打乱、不做增广（BN 在 eval 模式下用 running stats）。"""
        n = int(self.test_images.shape[0])
        for i in range(0, n, batch_size):
            index = slice(i, min(i + batch_size, n))
            yield self._batch(self.test_images, self.test_labels, index, False)
