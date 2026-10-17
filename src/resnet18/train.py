"""CIFAR-100 + ResNet18 训练脚本（CuPy 手写框架 nanonn）。

示例::

    python -m resnet18.train --epochs 100 --batch-size 128 --lr 0.1

针对 6GB 显存的笔记本 3060：

- 默认 ``--batch-size 128``，显存占用约 1.5~2GB（含 im2col 临时缓冲）；
  如果还想再快/再省显存，可以试 ``--batch-size 256`` / ``--batch-size 64``。
- 卷积的 im2col 分块上限由 ``nanonn.functional._MAX_IM2COL_BYTES`` 控制，
  显存极紧张时可以把模块里的常量调小。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cupy as cp

from nanonn import functional as F
from nanonn.optim import AdamW, CosineAnnealingLR, SGD, clip_grad_norm_

from .data import DEFAULT_CIFAR100_ROOT, CIFAR100
from .model import count_parameters, resnet18
from .utils import (
    AverageMeter,
    Timer,
    accuracy,
    evaluate,
    format_seconds,
    gpu_memory,
    human_bytes,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="用 CuPy 手写的 ResNet18 训练 CIFAR-100"
    )
    # ---- 数据 ----
    p.add_argument("--data-root", type=Path, default=DEFAULT_CIFAR100_ROOT,
                   help="CIFAR-100 根目录（默认 data/cifar100）")
    p.add_argument("--batch-size", type=int, default=128, help="batch 大小")
    p.add_argument("--no-augment", action="store_true", help="关闭随机裁剪/翻转")

    # ---- 优化 ----
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--clip-grad", type=float, default=5.0,
                   help="梯度全局范数裁剪阈值，<=0 表示关闭")

    # ---- 运行 ----
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", type=Path, default=None, help="从 .npz 权重继续训练")
    p.add_argument("--log-interval", type=int, default=50, help="每隔多少 step 打印一次")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="限制每 epoch 的 step 数（0 表示不限制，用于快速自测）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # 数据 / 模型 / 优化器
    # ------------------------------------------------------------------
    print(f"[数据] 载入 {args.data_root} ...")
    dataset = CIFAR100(args.data_root)
    print(
        f"[数据] 训练集 {dataset.train_images.shape[0]} 张，"
        f"测试集 {dataset.test_images.shape[0]} 张，类别数 {dataset.num_classes}"
    )

    model = resnet18(num_classes=dataset.num_classes)
    params = list(model.parameters())
    print(f"[模型] ResNet18-CIFAR，可训练参数 {count_parameters(model):,}")

    if args.optimizer == "sgd":
        optimizer = SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = CosineAnnealingLR(
        base_lr=args.lr,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )

    # ------------------------------------------------------------------
    # 可选：断点续训
    # ------------------------------------------------------------------
    start_epoch = 0
    best_acc = 0.0
    if args.resume is not None:
        saved_epoch, best_acc, _ = load_checkpoint(args.resume, model)
        best_acc = 0.0 if best_acc != best_acc else best_acc  # NaN -> 0
        start_epoch = saved_epoch + 1
        print(
            f"[续训] 从 {args.resume} 恢复，起点 epoch={start_epoch}，"
            f"best_acc={best_acc:.2%}"
        )

    steps_limit = args.steps_per_epoch if args.steps_per_epoch > 0 else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[训练] epochs={args.epochs} batch={args.batch_size} "
        f"optimizer={args.optimizer} lr={args.lr} wd={args.weight_decay}"
    )
    total_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        lr = scheduler.get_lr(epoch)
        optimizer.set_lr(lr)

        model.train()
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        epoch_timer = Timer()
        data_time = 0.0  # 累计取数据耗时，用于判断 CPU 端是否成为瓶颈
        last_data_ts = time.perf_counter()

        for step, (x, y) in enumerate(
            dataset.train_batches(
                args.batch_size, augment=not args.no_augment, shuffle=True
            )
        ):
            if steps_limit is not None and step >= steps_limit:
                break

            now = time.perf_counter()
            data_time += now - last_data_ts

            # ---- 前向 ----
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)

            # ---- 反向 + 参数更新 ----
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                clip_grad_norm_(params, args.clip_grad)
            optimizer.step()

            # ---- 统计 ----
            n = int(y.shape[0])
            loss_meter.update(float(loss.data), n)
            correct += accuracy(logits.data, y)
            total += n

            # 及时释放中间结果，让显存池可以复用（梯度已在 backward 中释放）
            del logits, loss, x, y

            last_data_ts = time.perf_counter()

            if (step + 1) % args.log_interval == 0:
                used, total_mem = gpu_memory()
                print(
                    f"  epoch {epoch:3d} step {step + 1:4d} "
                    f"loss {loss_meter.avg:.4f} acc {correct / total:.2%} "
                    f"lr {lr:.5f} "
                    f"显存 {human_bytes(used)}/{human_bytes(total_mem)} "
                    f"已用 {format_seconds(epoch_timer.elapsed())}",
                    flush=True,
                )

        train_loss = loss_meter.avg
        train_acc = correct / max(total, 1)

        # ------------------------------------------------------------------
        # 每个 epoch 结束后在测试集上评估
        # ------------------------------------------------------------------
        test_loss, test_acc = evaluate(
            model, dataset, batch_size=args.batch_size, label_smoothing=0.0
        )

        epoch_time = epoch_timer.elapsed()
        print(
            f"[{epoch + 1}/{args.epochs}] "
            f"lr={lr:.5f} train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2%} "
            f"time={format_seconds(epoch_time)} data={data_time:.1f}s",
            flush=True,
        )

        # ------------------------------------------------------------------
        # 保存权重：每轮都存 last，精度提升时存 best
        # ------------------------------------------------------------------
        extra = {
            "num_classes": dataset.num_classes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_acc": test_acc,
        }
        save_checkpoint(
            args.out_dir / "resnet18_last.npz", model, epoch, best_acc, extra
        )
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(
                args.out_dir / "resnet18_best.npz", model, epoch, best_acc, extra
            )
            print(
                f"  * 新的最佳精度 {best_acc:.2%}，已保存 resnet18_best.npz",
                flush=True,
            )

        # 归还显存池中空闲的块，避免碎片累积影响后续 epoch
        cp.get_default_memory_pool().free_all_blocks()

    total_time = time.perf_counter() - total_start
    print(
        f"[完成] 总耗时 {format_seconds(total_time)}，最佳测试精度 {best_acc:.2%}，"
        f"权重目录 {args.out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
