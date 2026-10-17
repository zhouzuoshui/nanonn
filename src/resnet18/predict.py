"""CIFAR-100 + ResNet18 推理/评估脚本。

用法::

    # 在测试集上评估
    python -m resnet18.predict --checkpoint checkpoints/resnet18_best.npz

    # 顺便打印若干随机样本的 top-5 预测
    python -m resnet18.predict --show 8

    # 预测一张自己的图片（需要 Pillow）
    python -m resnet18.predict --image path/to/cat.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cupy as cp
import numpy as np

from nanonn.tensor import Tensor, no_grad

from .data import DEFAULT_CIFAR100_ROOT, CIFAR100
from .model import resnet18
from .utils import evaluate, load_checkpoint

DEFAULT_CHECKPOINT = Path("checkpoints/resnet18_best.npz")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ResNet18 CIFAR-100 推理")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_CIFAR100_ROOT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--show", type=int, default=0,
                   help="打印多少个随机测试样本的 top-5 预测")
    p.add_argument("--image", type=Path, default=None,
                   help="预测单张图片（需要 Pillow）")
    return p.parse_args(argv)


def _preprocess(dataset: CIFAR100, images: np.ndarray) -> Tensor:
    """(N,3,32,32) uint8 -> 标准化后的 Tensor。"""
    x = cp.asarray(images, dtype=cp.float32)
    x *= 1.0 / 255.0
    x -= dataset.mean
    x /= dataset.std
    return Tensor(x, requires_grad=False)


def _logits(model, dataset: CIFAR100, images: np.ndarray) -> cp.ndarray:
    with no_grad():
        return model(_preprocess(dataset, images)).data


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    dataset = CIFAR100(args.data_root)
    model = resnet18(num_classes=dataset.num_classes)
    epoch, best_acc, extra = load_checkpoint(args.checkpoint, model)
    print(
        f"[权重] {args.checkpoint}（epoch={epoch}, 记录精度={best_acc:.2%}, {extra}）"
    )

    # ---- 测试集整体精度 ----
    test_loss, test_acc = evaluate(
        model, dataset, batch_size=args.batch_size, label_smoothing=0.0
    )
    print(f"[测试集] loss={test_loss:.4f} top-1 acc={test_acc:.2%}")

    # ---- 随机样本 top-5 ----
    if args.show > 0:
        n = int(dataset.test_images.shape[0])
        index = cp.random.choice(n, size=min(args.show, n), replace=False)
        images = cp.asnumpy(dataset.test_images[index])
        logits = _logits(model, dataset, images)
        probs = cp.asnumpy(cp.exp(logits - logits.max(axis=1, keepdims=True)))
        probs /= probs.sum(axis=1, keepdims=True)
        targets = cp.asnumpy(dataset.test_labels[index])

        print("\n[随机样本预测]")
        for i in range(images.shape[0]):
            top5 = np.argsort(-probs[i])[:5]
            preds = ", ".join(
                f"{dataset.label_names[j]}({probs[i, j]:.3f})" for j in top5
            )
            mark = "OK " if top5[0] == targets[i] else "ERR"
            print(
                f"  {mark} 真实={dataset.label_names[targets[i]]:<16} | {preds}"
            )

    # ---- 单张图片 ----
    if args.image is not None:
        try:
            from PIL import Image
        except ImportError:
            raise SystemExit("预测单张图片需要 Pillow：pip install pillow")

        img = Image.open(args.image).convert("RGB").resize((32, 32), Image.BILINEAR)
        images = np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)[None]  # 1,3,32,32
        logits = _logits(model, dataset, images)
        probs = cp.asnumpy(cp.exp(logits - logits.max(axis=1, keepdims=True)))[0]
        probs /= probs.sum()
        top5 = np.argsort(-probs)[:5]
        print(f"\n[图片] {args.image}")
        for j in top5:
            print(f"  {dataset.label_names[j]:<16} {probs[j]:.3f}")


if __name__ == "__main__":
    main()
