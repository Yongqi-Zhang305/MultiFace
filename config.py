"""
配置文件：所有超参数、路径、训练配置集中管理。
"""

import os
import torch

# ============================================================
# 路径配置
# ============================================================
# 项目根目录 (config.py 所在目录)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data", "UTKFace")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LOG_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据集配置
# ============================================================
IMG_SIZE = 224                # 输入图像尺寸 (224x224 for ResNet)
BATCH_SIZE = 128              # 批次大小 (增大以提升 GPU 利用率)
NUM_WORKERS = 4               # DataLoader 进程数 (SubsetDataset 已模块级, 可用多进程)
TRAIN_RATIO = 0.70            # 训练集比例
VAL_RATIO = 0.15              # 验证集比例
TEST_RATIO = 0.15             # 测试集比例

# 年龄分箱 (用于年龄分类准确率统计)
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]

# ============================================================
# 模型配置
# ============================================================
BACKBONE = "resnet18"         # 骨干网络: resnet18 / resnet34 / resnet50 / mobilenet_v2
PRETRAINED = True             # 是否使用 ImageNet 预训练权重
GENDER_CLASSES = 2            # 性别类别数 (男/女)
NUM_AGE_CLASSES = 117         # 年龄类别数 (0-116岁, 完全覆盖UTKFace数据集范围)
AGE_CE_LAMBDA = 0.5           # CE+MAE联合损失中MAE的权重

# ============================================================
# 训练配置
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 60                   # 训练总轮数 (对于该数据集通常已足够收敛)
WEIGHT_DECAY = 1e-4           # 权重衰减 (L2 正则化)

# 分阶段训练 + 不同头不同学习率
STAGE_SPLIT = 0.5             # 前 50% epoch 冻结性别头, 只训年龄头+骨干
BACKBONE_LR = 1e-5            # 骨干网络学习率 (预训练权重, 用较小LR微调)
GENDER_HEAD_LR = 1e-4         # 性别头学习率
AGE_HEAD_LR = 2e-4            # 年龄头学习率 (从头训练, 用较大LR)

LR_ETA_MIN = 1e-7             # 余弦退火最小学习率
EARLY_STOP_PATIENCE = 10      # 早停耐心值 (验证 loss 不降则停止)

# ============================================================
# 打印配置摘要
# ============================================================
def print_config():
    print("=" * 60)
    print("   MultiFace — 人脸性别与年龄预测项目")
    print("=" * 60)
    print(f"  数据目录:     {DATA_DIR}")
    print(f"  图像尺寸:     {IMG_SIZE}×{IMG_SIZE}")
    print(f"  批次大小:     {BATCH_SIZE}")
    print(f"  骨干网络:     {BACKBONE} (pretrained={PRETRAINED})")
    print(f"  年龄类别数:   {NUM_AGE_CLASSES} (0-{NUM_AGE_CLASSES-1}岁)")
    print(f"  年龄损失:     CE + {AGE_CE_LAMBDA}×MAE")
    print(f"  训练设备:     {DEVICE}")
    print(f"  训练轮数:     {EPOCHS}")
    print(f"  骨干网络 LR:  {BACKBONE_LR}")
    print(f"  性别头 LR:    {GENDER_HEAD_LR}")
    print(f"  年龄头 LR:    {AGE_HEAD_LR}")
    print(f"  分阶段训练:   前{int(STAGE_SPLIT*100)}%冻结性别头")
    print(f"  输出目录:     {OUTPUT_DIR}")
    print("=" * 60)
