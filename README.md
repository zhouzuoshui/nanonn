# nanonn

用 **CuPy 从零手写** 的极小深度学习框架，并在 CIFAR-100 上训练 ResNet18。
全程不使用 PyTorch（也不使用任何其他深度学习框架）：自动求导、卷积、批归一化、
优化器、数据增广全部自己实现。

目标硬件是一台 6GB 显存的笔记本（RTX 3060 Laptop），默认超参与显存控制都按这个
预算调整。

## 目录结构

```
src/
├── nanonn/              # 框架本体（纯 CuPy 实现）
│   ├── tensor.py        # Tensor / Function / 反向传播引擎 / no_grad
│   ├── functional.py    # 算子：im2col 卷积、BatchNorm、MaxPool、CrossEntropy 等
│   ├── nn.py            # Module 系统与常用层（Linear/Conv2d/BatchNorm2d/...）
│   └── optim.py         # SGD / AdamW / 余弦退火 / 梯度裁剪
└── resnet18/            # CIFAR-100 训练与推理
    ├── model.py         # CIFAR 版 ResNet18（3x3 stem，无 maxpool）
    ├── data.py          # 直接读 tar.gz + GPU 端随机裁剪/翻转
    ├── train.py         # 训练入口
    ├── predict.py       # 评估 / 单图推理
    ├── gradcheck.py     # 数值梯度自检（手写算子的回归测试）
    └── utils.py         # 计时、精度、权重存取等工具
data/cifar100/raw/cifar-100-python.tar.gz   # 原始数据（已就位）
```

## 环境

依赖由 `uv` 管理（`pyproject.toml` + `uv.lock`）：

```bash
uv sync            # 按 lock 文件安装
```

依赖包含 `cupy-cuda13x`、`numpy`，以及 CuPy 运行时需要的 CUDA 组件
`cuda-toolkit[cublas,cudart,curand,nvrtc]`（约 700MB，提供 cublas 矩阵乘、
curand 随机数、以及 elementwise kernel 需要的头文件）。

## 用法

所有命令都在项目根目录执行。

```bash
# 训练（默认 100 epoch，SGD + 余弦退火 + warmup，batch 128）
uv run python -m resnet18.train

# 少步数快速验证流程是否能跑通
uv run python -m resnet18.train --epochs 1 --steps-per-epoch 5

# 换优化器 / 调超参
uv run python -m resnet18.train --optimizer adamw --lr 1e-3 --epochs 60

# 断点续训
uv run python -m resnet18.train --resume checkpoints/resnet18_last.npz
```

训练过程会在 `checkpoints/` 下保存 `resnet18_last.npz`（最新）与
`resnet18_best.npz`（测试集精度最高）。

```bash
# 在测试集上评估
uv run python -m resnet18.predict --checkpoint checkpoints/resnet18_best.npz

# 顺便打印 8 个随机样本的 top-5 预测
uv run python -m resnet18.predict --show 8

# 预测自己的图片（需要额外装 Pillow）
uv run python -m resnet18.predict --image cat.png
```

### 算子自检

改动 `nanonn/functional.py` 里的任何算子后，先跑一遍数值梯度校验：

```bash
uv run python -m resnet18.gradcheck
```

它会对 20 个张量输入做 float64 中心差分，与手写 `backward` 对比相对误差，
全部通过时打印 `ALL PASS` 并以退出码 0 结束（可直接接进 CI）。
覆盖的分支：卷积的 stride=1/2、padding=0/1、以及 `(oh-1)*stride+kh` 小于
「输入+padding」的 ragged 情况、im2col 分块路径、BatchNorm、MaxPool、
CrossEntropy（label smoothing 0/0.1）、matmul + 广播 bias。

## 6GB 显存调参建议

| 项目 | 建议 |
| --- | --- |
| `--batch-size` | 128 起步；显存不足降到 64（此时把 `--lr` 同比例下调） |
| `--clip-grad` | 默认 5.0，一般无需改动 |
| 卷积 im2col 缓冲 | `nanonn.functional._MAX_IM2COL_BYTES`（默认 64MB），显存吃紧可降到 32MB |
| 显存回收 | 每个 epoch 结束会 `free_all_blocks()`，日志里会打印实时占用 |

## 实现要点

- **自动求导**：eager 模式，`Tensor.backward()` 做迭代式拓扑排序；非叶子节点的梯度
  传播后立即释放，显著降低显存峰值。
- **卷积**：手写 im2col + 矩阵乘，按样本分块（`_MAX_IM2COL_BYTES` 控制块大小），
  反向传播用 col2im 散射累加。
- **BatchNorm**：训练用 batch 统计量（有偏方差归一化），running_var 用无偏方差更新，
  与 PyTorch 行为一致。
- **数据增广**：图像以 uint8 常驻显存（训练集 150MB、测试集 30MB），
  随机裁剪 + 随机水平翻转完全在 GPU 上向量化完成，无 DataLoader 线程。
- **梯度正确性**：所有算子都用 float64 中心差分数值梯度校验过（相对误差 < 1e-9），
  覆盖 stride/padding 组合、分块路径、label smoothing 等分支；
  随时可用 `uv run python -m resnet18.gradcheck` 复验。
