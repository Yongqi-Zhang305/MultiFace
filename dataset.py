"""
数据集模块：加载 UTKFace 数据集，解析文件名获取标签，划分训练/验证/测试集。
文件名格式: [age]_[gender]_[race]_[timestamp].jpg.chip.jpg
  - age:    0~116 岁
  - gender: 0=男, 1=女
  - race:   0=白人, 1=黑人, 2=亚裔, 3=印度裔, 4=其他
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

from config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
)


class SubsetDataset(Dataset):
    """从已有 Dataset 和索引列表创建子集，并应用指定 transform。
    必须定义在模块级别，以确保 multiprocessing pickle 可用。
    """

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        path = self.base.file_paths[original_idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        age = torch.tensor(self.base.ages[original_idx], dtype=torch.float32)
        gender = torch.tensor(self.base.genders[original_idx], dtype=torch.long)
        return image, gender, age


class UTKFaceDataset(Dataset):
    """UTKFace 自定义数据集"""

    def __init__(self, file_paths, transform=None):
        """
        Args:
            file_paths: 图片文件路径列表
            transform: torchvision 图像变换
        """
        self.file_paths = file_paths
        self.transform = transform

        # 预解析所有标签
        self.ages = []
        self.genders = []
        self.races = []

        for path in file_paths:
            filename = os.path.basename(path)
            parts = filename.split("_")
            if len(parts) < 3:
                # 跳过命名不规范的图片
                self.ages.append(-1)
                self.genders.append(-1)
                self.races.append(-1)
                continue

            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2]) if len(parts) >= 3 else -1
            except ValueError:
                age, gender, race = -1, -1, -1

            self.ages.append(age)
            self.genders.append(gender)
            self.races.append(race)

        # 过滤无效数据
        valid_indices = [
            i for i, (a, g) in enumerate(zip(self.ages, self.genders))
            if a >= 0 and g in (0, 1)
        ]
        self.file_paths = [self.file_paths[i] for i in valid_indices]
        self.ages = [self.ages[i] for i in valid_indices]
        self.genders = [self.genders[i] for i in valid_indices]
        self.races = [self.races[i] for i in valid_indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.long)

        return image, gender, age

    def get_stats(self):
        """返回数据集的统计信息"""
        ages = np.array(self.ages)
        genders = np.array(self.genders)
        return {
            "total": len(self),
            "male": int((genders == 0).sum()),
            "female": int((genders == 1).sum()),
            "age_mean": float(ages.mean()),
            "age_std": float(ages.std()),
            "age_min": int(ages.min()),
            "age_max": int(ages.max()),
        }


def get_transforms(is_train=True):
    """获取数据变换管道

    Args:
        is_train: True 返回训练用增强变换, False 返回测试用基础变换

    训练时加入数据增强提升泛化能力:
        - 随机水平翻转 (人脸左右对称，翻转不影响标签)
        - 随机旋转 (±15°)
        - 颜色抖动 (亮度/对比度/饱和度微调)
        - 随机仿射变换 (模拟不同角度)
    """
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform


def create_dataloaders():
    """创建训练/验证/测试 DataLoader

    Returns:
        train_loader, val_loader, test_loader, dataset_stats
    """
    # 收集所有图片路径
    all_files = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".jpeg"):
            all_files.append(os.path.join(DATA_DIR, fname))

    print(f"\n[数据集] 发现 {len(all_files)} 个图片文件")

    # 先用基础 transform 创建完整数据集以获取统计信息
    full_dataset = UTKFaceDataset(all_files, transform=None)
    stats = full_dataset.get_stats()

    print(f"[数据集] 有效样本: {stats['total']}")
    print(f"[数据集] 男性: {stats['male']}, 女性: {stats['female']}")
    print(f"[数据集] 年龄: 均值={stats['age_mean']:.1f}, 标准差={stats['age_std']:.1f}, "
          f"范围=[{stats['age_min']}, {stats['age_max']}]")

    # 按比例划分数据集 (先确定划分，再分别应用不同的 transform)
    total = len(full_dataset)
    train_size = int(total * TRAIN_RATIO)
    val_size = int(total * VAL_RATIO)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size],
        generator=generator
    )

    # 为不同子集创建带 transform 的 Dataset
    train_dataset = SubsetDataset(full_dataset, train_indices, get_transforms(is_train=True))
    val_dataset = SubsetDataset(full_dataset, val_indices, get_transforms(is_train=False))
    test_dataset = SubsetDataset(full_dataset, test_indices, get_transforms(is_train=False))

    print(f"[数据集] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, "
          f"测试集: {len(test_dataset)}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    # 快速自检：加载数据并打印统计
    train_loader, val_loader, test_loader, stats = create_dataloaders()

    images, genders, ages = next(iter(train_loader))
    print(f"\n[检查] 一个 batch 的形状: images={images.shape}, genders={genders.shape}, ages={ages.shape}")
    print(f"[检查] 性别分布: {genders.bincount()}")
    print(f"[检查] 年龄范围: [{ages.min():.0f}, {ages.max():.0f}]")
