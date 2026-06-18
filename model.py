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
 │ FC 2     │      │  FC 1      │
 │ Softmax  │      │  (回归值)   │
 └─────────┘      └───────────┘
  Gender cls           Age reg
"""

import torch
import torch.nn as nn
from torchvision import models

from config import GENDER_CLASSES, BACKBONE, PRETRAINED


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

        # ── 年龄回归头 ──
        self.age_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
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
            gender_out: (B, 2) 性别分类 logits
            age_out:    (B, 1) 年龄回归值
        """
        features = self.backbone(x)            # (B, feature_dim)  — Identity 已替代 FC
        gender_out = self.gender_head(features)
        age_out = self.age_head(features)
        return gender_out, age_out


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
    """返回两个任务的损失函数"""
    gender_criterion = nn.CrossEntropyLoss()
    age_criterion = nn.L1Loss()
    return gender_criterion, age_criterion


if __name__ == "__main__":
    model = build_model()
    dummy_input = torch.randn(4, 3, 224, 224).to(next(model.parameters()).device)
    gender_out, age_out = model(dummy_input)
    print(f"[检查] 输入形状: {dummy_input.shape}")
    print(f"[检查] 性别输出: {gender_out.shape}  (期望: [4, 2])")
    print(f"[检查] 年龄输出: {age_out.shape}  (期望: [4, 1])")
