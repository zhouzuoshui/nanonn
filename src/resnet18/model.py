"""用 nanonn（CuPy 手写）实现的 ResNet18，适配 CIFAR-100 的 32x32 输入。

与 ImageNet 版本的区别（CIFAR 版的标准做法）：

1. stem 用 3x3/stride=1 卷积代替 7x7/stride=2，并且去掉 maxpool，
   这样 32x32 的图在网络入口不会被过度下采样；
2. 四个 stage 的下采样只发生在 stage2/3/4 的第一个 block（stride=2），
   特征图尺寸依次为 32 -> 32 -> 16 -> 8 -> 4；
3. 最后用全局平均池化 + 全连接层输出 100 类。

BasicBlock 结构::

    x --> conv3x3 --> BN --> ReLU --> conv3x3 --> BN --> (+) --> ReLU --> out
          |                                                |
          +------------------ shortcut -------------------+

当输入输出通道数不一致、或需要下采样时，shortcut 换成一个
1x1 卷积 + BN（即 downsample 分支）。
"""

from __future__ import annotations

from nanonn import functional as F
from nanonn.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    Linear,
    Module,
    Sequential,
    constant_,
    kaiming_normal_,
)
from nanonn.tensor import Tensor

__all__ = ["BasicBlock", "ResNet18", "resnet18", "count_parameters"]


class BasicBlock(Module):
    """ResNet 的基础残差块（两层 3x3 卷积 + 恒等/投影 shortcut）。"""

    expansion = 1  # 输出通道相对中间通道的倍率，BasicBlock 为 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 第一个 3x3 卷积承担空间下采样（stride 由外部传入）
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        # 第二个 3x3 卷积保持分辨率
        self.conv2 = Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        # 只有形状不匹配时才需要投影分支；显式赋 None 不会注册为子模块
        if stride != 1 or in_channels != out_channels:
            self.downsample: Module | None = Sequential(
                Conv2d(in_channels, out_channels, 1, stride=stride, padding=0),
                BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 投影分支与主分支在通道/尺寸上对齐后再相加
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return F.relu(out)


class ResNet18(Module):
    """面向 CIFAR-100（32x32 输入、100 类）的 ResNet18。"""

    def __init__(self, num_classes: int = 100, in_channels: int = 3):
        super().__init__()
        # ---- stem：3x3 卷积，不下采样 ----
        self.conv1 = Conv2d(in_channels, 64, 3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(64)

        # ---- 四个 stage，每个 stage 2 个 BasicBlock ----
        # stage1: 32x32,  64 通道；后三个 stage 各自在第一个 block 下采样一半
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # ---- 分类头 ----
        self.avgpool = AdaptiveAvgPool2d(1)
        self.fc = Linear(512, num_classes)

        self._init_weights()

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> Sequential:
        """构造一个 stage。只有第一个 block 使用给定 stride 做下采样。"""
        layers = [BasicBlock(in_channels, out_channels, stride)]
        # 后续 block 输入输出通道与分辨率都不变
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming 正态初始化（与 torchvision 的 ResNet 保持一致）。

        卷积层使用 fan_out（out_channels * k * k），全连接层使用 fan_in。
        BatchNorm 的 weight/bias 在层内部已初始化为 1/0。
        """
        for m in self.modules():
            if isinstance(m, Conv2d):
                fan_out = m.out_channels * m.kernel_size * m.kernel_size
                kaiming_normal_(m.weight, fan_out)
            elif isinstance(m, Linear):
                kaiming_normal_(m.weight, m.in_features)
                if m.bias is not None:
                    constant_(m.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)  # 32x32,  64
        x = self.layer2(x)  # 16x16, 128
        x = self.layer3(x)  #  8x8,  256
        x = self.layer4(x)  #  4x4,  512

        x = self.avgpool(x)  # (N, 512)
        x = self.fc(x)       # (N, num_classes)
        return x


def resnet18(num_classes: int = 100) -> ResNet18:
    """工厂函数，方便以后扩展到 resnet34 等变体。"""
    return ResNet18(num_classes=num_classes)


def count_parameters(model: Module) -> int:
    """统计可训练参数总量（用于日志中确认模型规模）。"""
    return sum(int(p.data.size) for p in model.parameters())
