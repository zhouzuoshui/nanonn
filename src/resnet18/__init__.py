"""CIFAR-100 + ResNet18 的训练/推理代码（基于手写的 nanonn 框架）。

- :mod:`resnet18.model`   ResNet18 网络结构（CIFAR 版）
- :mod:`resnet18.data`    CIFAR-100 数据加载与 GPU 端增广
- :mod:`resnet18.train`   训练入口
- :mod:`resnet18.predict` 测试集评估与推理
"""

from .model import BasicBlock, ResNet18, resnet18

__all__ = ["BasicBlock", "ResNet18", "resnet18"]
