# MultiFace — 基于人脸图像的性别与年龄预测

基于 PyTorch 的多任务深度学习项目，输入一张人脸图片，同时预测**性别**（男/女）和**年龄**（连续值）。使用 ResNet 骨干网络 + 双输出头架构，端到端训练。

---

## 项目结构

```
MultiFace/
├── main.py              # 程序入口，训练主流程
├── train.py             # 训练/验证/评估核心函数
├── model.py             # 多任务模型定义
├── dataset.py           # 数据加载与预处理
├── config.py            # 超参数与路径配置
├── requirements.txt     # Python 依赖
├── README.md            # 本文件
├── data/
│   └── UTKFace/         # 数据集（23,708 张图片）
└── output/
    ├── best_model.pth   # 训练保存的最佳模型
    └── training_log.csv # 每轮训练指标日志
```

---

## 数据集

使用 **UTKFace** 数据集，包含 23,708 张已对齐裁剪的人脸图片。

- **年龄范围**: 0 – 116 岁
- **性别**: 0 = 男, 1 = 女（约 52% 男 / 48% 女）
- **种族**: 0 = 白人, 1 = 黑人, 2 = 亚裔, 3 = 印度裔, 4 = 其他
- **图片格式**: 224×224 已对齐裁剪

文件名格式：`[age]_[gender]_[race]_[timestamp].jpg.chip.jpg`

> 数据来源：[UTKFace on Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new)

---

## 环境安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# 安装依赖
pip install -r requirements.txt
```

依赖清单：

| 包 | 最低版本 |
|---|---|
| `torch` | ≥ 2.0.0 |
| `torchvision` | ≥ 0.15.0 |
| `Pillow` | ≥ 9.0.0 |
| `numpy` | ≥ 1.21.0 |

---

## 使用方法

### 训练模型

```bash
python main.py
```

训练过程会：
- 每个 epoch 输出训练集和验证集的**损失函数值**与**准确率**
- 自动保存验证损失最小的模型到 `output/best_model.pth`
- 训练日志写入 `output/training_log.csv`
- 若验证损失连续 10 轮不下降则触发**早停**

### 检查数据

```bash
python dataset.py
```

### 检查模型结构

```bash
python model.py
```

---

## 模型架构

```
┌─────────────────────┐
│   输入图像 (224×224)  │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   ResNet18 Backbone │  ← ImageNet 预训练，去掉最后 FC
│   (共享特征提取)      │
└──────────┬──────────┘
           │
  ┌────────┴────────┐
  │                 │
┌─▼────────┐  ┌─────▼──────┐
│ 性别分类头 │  │  年龄回归头  │
│ FC 256    │  │  FC 256     │
│ BN + ReLU │  │  BN + ReLU  │
│ Dropout   │  │  Dropout    │
│ FC 128    │  │  FC 128     │
│ BN + ReLU │  │  BN + ReLU  │
│ Dropout   │  │  Dropout    │
│ FC 2      │  │  FC 1       │
│ → 男/女   │  │  → 年龄值    │
└──────────┘  └────────────┘
```

- **骨干网络**: ResNet18（ImageNet 预训练），可更换为 ResNet34/50 或 MobileNetV2
- **性别任务**: 二分类，CrossEntropyLoss
- **年龄任务**: 回归，L1Loss（MAE）
- **总损失**: `Loss = GenderLoss + 0.4 × AgeLoss`
- **优化器**: AdamW
- **学习率调度**: StepLR（每 15 epoch × 0.5）

---

## 训练指标

每个 epoch 输出以下指标（训练集 + 验证集）：

| 指标 | 说明 |
|------|------|
| **Gender Loss** | 性别分类交叉熵损失 |
| **Age Loss** | 年龄回归 L1 损失 |
| **Total Loss** | 加权总损失 |
| **Gender Acc** | 性别分类准确率 (%) |
| **Age MAE** | 年龄预测平均绝对误差 (年) |
| **Age RMSE** | 年龄预测均方根误差 (年) |
| **Age Bin Acc** | 年龄段分箱准确率 (%) — 预测年龄与真实年龄落入同一区间 |

年龄分箱边界：`[0, 10, 20, 30, 40, 50, 60, 70, 120]`

---

## 配置说明

所有可调参数集中在 [`config.py`](config.py)：

```python
# 路径
DATA_DIR          # 数据集目录（相对路径，跨平台）
OUTPUT_DIR        # 输出目录

# 数据
IMG_SIZE = 224    # 输入图像尺寸
BATCH_SIZE = 128  # 批次大小
NUM_WORKERS = 4   # DataLoader 并行进程数

# 模型
BACKBONE = "resnet18"   # 骨干网络：resnet18 / resnet34 / resnet50 / mobilenet_v2
PRETRAINED = True       # 是否使用预训练权重
AGE_LOSS_WEIGHT = 0.4   # 年龄损失占总损失的权重

# 训练
EPOCHS = 30             # 最大训练轮数
LEARNING_RATE = 1e-4    # 学习率
WEIGHT_DECAY = 1e-4     # 权重衰减
EARLY_STOP_PATIENCE = 10  # 早停耐心值
```

---

## 跨平台支持

所有路径基于项目根目录动态计算（`config.py` 中的 `ROOT_DIR`），无论在 Windows / Linux / macOS 上，只要保持目录结构不变，直接运行即可。

---

## License

本项目仅用于学习与研究目的。UTKFace 数据集请遵循其原始许可协议。
