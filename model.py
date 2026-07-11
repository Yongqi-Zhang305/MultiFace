"""
模型模块：多任务学习模型 —— 共享骨干网络 + 性别分类头 + 年龄回归头。

架构设计:
    ┌─────────────────────┐
    │   输入图像 (224×224)  │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   ResNet Backbone   │  ← 共享特征提取 (ImageNet 预训练)
    │  (去掉最后的 FC 层)   │
    └──────────┬──────────┘
               │
      ┌────────┴────────┐
      │                 │
 ┌────▼────┐      ┌─────▼─────┐
 │ 性别分支  │      │  年龄分支   │
 │ FC 256   │      │  FC 256    │
 │ ReLU     │      │  ReLU      │
 │ Dropout  │      │  Dropout   │
 │ FC 2     │      │  FC 101    │
 │ Softmax  │      │  Softmax   │
 └─────────┘      └───────────┘
  Gender cls        Age 101 cls
                 → 期望年龄 (加权求和)
"""

import torch
import torch.nn as nn
from torchvision import models

from config import GENDER_CLASSES, NUM_AGE_CLASSES, BACKBONE, PRETRAINED


class MultiTaskModel(nn.Module):
    """多任务学习模型：将预训练 CNN 的最后一层替换为双输出头"""

    def __init__(self, backbone_name=BACKBONE, pretrained=PRETRAINED):
        super().__init__()

        # ── 构建骨干网络 (去掉最后的 FC/Classifier 层) ──
        self.backbone, self.feature_dim = self._build_backbone(backbone_name, pretrained)

        # ── 性别分类头 ──
        self.gender_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, GENDER_CLASSES),
        )

        # ── 年龄分类头 (101 类 Softmax → 期望年龄) ──
        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, NUM_AGE_CLASSES),  # 输出 0-100 岁每类的分数
        )

        self._init_heads()

    def _build_backbone(self, name, pretrained):
        """构建骨干网络并将最后的分类层替换为 Identity。

        Returns:
            backbone: nn.Module, 输出特征向量 (未 flatten)
            feature_dim: int, 特征维度
        """
        if name == "resnet18":
            model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
            model.fc = nn.Identity()
        elif name == "resnet34":
            model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
            model.fc = nn.Identity()
        elif name == "resnet50":
            model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 2048
            model.fc = nn.Identity()
        elif name == "mobilenet_v2":
            model = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 1280
            model.classifier = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {name}。可选: resnet18, resnet34, resnet50, mobilenet_v2")

        return model, feature_dim

    def _init_heads(self):
        """对新增的全连接层进行 Kaiming 初始化"""
        for head in [self.gender_head, self.age_head]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """前向传播

        Args:
            x: 输入图像 tensor, shape (B, 3, H, W)

        Returns:
            gender_out:  (B, 2)   性别分类 logits
            age_logits:  (B, 101) 年龄分类 logits (0-100岁)
        """
        features = self.backbone(x)            # (B, feature_dim)
        gender_out = self.gender_head(features)
        age_logits = self.age_head(features)   # (B, NUM_AGE_CLASSES)
        return gender_out, age_logits


def compute_expected_age(age_logits):
    """从 101 类 Softmax 概率计算期望年龄 (DEX 方法)

    Args:
        age_logits: (B, NUM_AGE_CLASSES) 年龄分类 logits

    Returns:
        expected_age: (B,) 年龄期望值 = sum(softmax(logits)[i] * i)
    """
    probs = torch.softmax(age_logits, dim=1)  # (B, 101)
    indices = torch.arange(
        age_logits.size(1), dtype=torch.float32, device=age_logits.device
    )  # [0, 1, 2, ..., 100]
    return (probs * indices).sum(dim=1)       # (B,)


def build_model():
    """工厂函数：创建模型并移动到设备，打印参数量"""
    from config import DEVICE

    model = MultiTaskModel()
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型] 骨干网络: {BACKBONE}")
    print(f"[模型] 总参数量: {total_params:,}")
    print(f"[模型] 可训练参数: {trainable_params:,}")

    return model


def get_loss_functions():
    """返回损失函数 (性别分类器用 CrossEntropy, 年龄在 train.py 中手动计算 CE+MAE)"""
    gender_criterion = nn.CrossEntropyLoss()
    return gender_criterion


if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.randn(4, 3, 224, 224).to(next(model.parameters()).device)
    gender_out, age_logits = model(dummy_input)
    expected_age = compute_expected_age(age_logits)
    print(f"[检查] 输入形状:   {dummy_input.shape}")
    print(f"[检查] 性别输出:   {gender_out.shape}   (期望: [4, 2])")
    print(f"[检查] 年龄 logits: {age_logits.shape} (期望: [4, {NUM_AGE_CLASSES}])")
    print(f"[检查] 期望年龄:   {expected_age}")
