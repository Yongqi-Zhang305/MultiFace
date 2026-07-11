"""
训练核心模块：提供训练/验证/评估函数，供 main.py 调用。
年龄使用 101 类 Softmax 分类 + 期望回归 (DEX)，损失函数 = CE + λ×MAE。
"""

import os
import csv
import numpy as np

import torch
import torch.nn.functional as F

from config import (
    DEVICE, NUM_AGE_CLASSES, AGE_CE_LAMBDA, LOG_PATH, AGE_BINS,
)
from model import compute_expected_age


# ============================================================
# 辅助函数
# ============================================================

def compute_age_bin_accuracy(pred_ages, true_ages, bins=AGE_BINS):
    """
    计算年龄分箱准确率：预测年龄和真实年龄落入同一区间的比例。

    Args:
        pred_ages: np.array, shape (N,) 预测年龄 (期望年龄, 浮点数)
        true_ages: np.array, shape (N,) 真实年龄
        bins: 分箱边界

    Returns:
        accuracy: float
    """
    pred_bins = np.digitize(pred_ages, bins) - 1
    true_bins = np.digitize(true_ages, bins) - 1
    pred_bins = np.clip(pred_bins, 0, len(bins) - 1)
    true_bins = np.clip(true_bins, 0, len(bins) - 1)
    return (pred_bins == true_bins).mean()


def format_time(seconds):
    """将秒数格式化为 mm:ss"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ============================================================
# 训练一个 epoch
# ============================================================

def train_one_epoch(model, dataloader, gender_criterion, optimizer, epoch):
    """训练一个 epoch

    年龄损失 = CrossEntropyLoss(101类) + AGE_CE_LAMBDA × L1Loss(期望年龄, 真实年龄)

    Returns:
        metrics: dict, 包含各种训练指标的均值
    """
    model.train()

    total_gender_loss = 0.0
    total_age_ce = 0.0
    total_age_mae = 0.0
    total_age_loss = 0.0
    total_loss = 0.0
    correct_gender = 0
    total_samples = 0
    all_pred_ages = []
    all_true_ages = []

    for batch_idx, (images, genders, ages) in enumerate(dataloader):
        images = images.to(DEVICE)
        genders = genders.to(DEVICE)
        ages = ages.to(DEVICE).float()  # (B,)

        # 年龄标签：分类用 (整数 0-100)，MAE 用 (原始浮点值)
        age_class_labels = ages.round().long().clamp(0, NUM_AGE_CLASSES - 1)  # (B,)

        # ── 前向传播 ──
        gender_logits, age_logits = model(images)  # (B,2), (B,101)

        # ── 性别损失 ──
        loss_g = gender_criterion(gender_logits, genders)

        # ── 年龄损失: CE + λ × MAE ──
        loss_age_ce = F.cross_entropy(age_logits, age_class_labels)
        expected_age = compute_expected_age(age_logits)            # (B,)
        loss_age_mae = F.l1_loss(expected_age, ages)               # 标量
        loss_a = loss_age_ce + AGE_CE_LAMBDA * loss_age_mae

        # ── 总损失 ──
        loss = loss_g + loss_a

        # ── 反向传播 ──
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── 统计 ──
        bs = images.size(0)
        total_samples += bs
        total_gender_loss += loss_g.item() * bs
        total_age_ce += loss_age_ce.item() * bs
        total_age_mae += loss_age_mae.item() * bs
        total_age_loss += loss_a.item() * bs
        total_loss += loss.item() * bs

        # 性别准确率
        _, predicted_gender = torch.max(gender_logits, 1)
        correct_gender += (predicted_gender == genders).sum().item()

        # 收集期望年龄用于后续 MAE/RMSE/分箱准确率计算
        all_pred_ages.extend(expected_age.detach().cpu().numpy().flatten())
        all_true_ages.extend(ages.detach().cpu().numpy().flatten())

        # 每 100 个 batch 打印一次进度
        if batch_idx % 100 == 0 and batch_idx > 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    Batch [{batch_idx:4d}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(G: {loss_g.item():.4f}, ACE: {loss_age_ce.item():.4f}, "
                  f"AMAE: {loss_age_mae.item():.4f}) "
                  f"LR: {current_lr:.2e}")

    # ── 计算 epoch 平均指标 ──
    avg_gender_loss = total_gender_loss / total_samples
    avg_age_ce = total_age_ce / total_samples
    avg_age_mae = total_age_mae / total_samples
    avg_age_loss = total_age_loss / total_samples
    avg_total_loss = total_loss / total_samples
    gender_acc = correct_gender / total_samples * 100

    all_pred_ages = np.array(all_pred_ages)
    all_true_ages = np.array(all_true_ages)
    age_mae = np.abs(all_pred_ages - all_true_ages).mean()
    age_rmse = np.sqrt(((all_pred_ages - all_true_ages) ** 2).mean())
    age_bin_acc = compute_age_bin_accuracy(all_pred_ages, all_true_ages) * 100

    return {
        "gender_loss": avg_gender_loss,
        "age_ce": avg_age_ce,
        "age_mae_loss": avg_age_mae,
        "age_loss": avg_age_loss,
        "total_loss": avg_total_loss,
        "gender_acc": gender_acc,
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_bin_acc": age_bin_acc,
    }


# ============================================================
# 验证/测试一个 epoch
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, gender_criterion):
    """评估模型（验证或测试）"""
    model.eval()

    total_gender_loss = 0.0
    total_age_ce = 0.0
    total_age_mae = 0.0
    total_age_loss = 0.0
    total_loss = 0.0
    correct_gender = 0
    total_samples = 0
    all_pred_ages = []
    all_true_ages = []

    for images, genders, ages in dataloader:
        images = images.to(DEVICE)
        genders = genders.to(DEVICE)
        ages = ages.to(DEVICE).float()  # (B,)

        age_class_labels = ages.round().long().clamp(0, NUM_AGE_CLASSES - 1)

        gender_logits, age_logits = model(images)

        loss_g = gender_criterion(gender_logits, genders)
        loss_age_ce = F.cross_entropy(age_logits, age_class_labels)
        expected_age = compute_expected_age(age_logits)
        loss_age_mae = F.l1_loss(expected_age, ages)
        loss_a = loss_age_ce + AGE_CE_LAMBDA * loss_age_mae
        loss = loss_g + loss_a

        bs = images.size(0)
        total_samples += bs
        total_gender_loss += loss_g.item() * bs
        total_age_ce += loss_age_ce.item() * bs
        total_age_mae += loss_age_mae.item() * bs
        total_age_loss += loss_a.item() * bs
        total_loss += loss.item() * bs

        _, predicted_gender = torch.max(gender_logits, 1)
        correct_gender += (predicted_gender == genders).sum().item()

        all_pred_ages.extend(expected_age.cpu().numpy().flatten())
        all_true_ages.extend(ages.cpu().numpy().flatten())

    avg_gender_loss = total_gender_loss / total_samples
    avg_age_ce = total_age_ce / total_samples
    avg_age_mae = total_age_mae / total_samples
    avg_age_loss = total_age_loss / total_samples
    avg_total_loss = total_loss / total_samples
    gender_acc = correct_gender / total_samples * 100

    all_pred_ages = np.array(all_pred_ages)
    all_true_ages = np.array(all_true_ages)
    age_mae = np.abs(all_pred_ages - all_true_ages).mean()
    age_rmse = np.sqrt(((all_pred_ages - all_true_ages) ** 2).mean())
    age_bin_acc = compute_age_bin_accuracy(all_pred_ages, all_true_ages) * 100

    return {
        "gender_loss": avg_gender_loss,
        "age_ce": avg_age_ce,
        "age_mae_loss": avg_age_mae,
        "age_loss": avg_age_loss,
        "total_loss": avg_total_loss,
        "gender_acc": gender_acc,
        "age_mae": age_mae,
        "age_rmse": age_rmse,
        "age_bin_acc": age_bin_acc,
    }


# ============================================================
# 日志记录
# ============================================================

def log_epoch(epoch, train_metrics, val_metrics, lr, elapsed):
    """将每个 epoch 的指标写入 CSV 日志并打印到控制台"""
    row = {
        "epoch": epoch,
        "lr": lr,
        "train_gender_loss": f"{train_metrics['gender_loss']:.6f}",
        "train_age_ce": f"{train_metrics['age_ce']:.6f}",
        "train_age_mae_loss": f"{train_metrics['age_mae_loss']:.6f}",
        "train_age_loss": f"{train_metrics['age_loss']:.6f}",
        "train_total_loss": f"{train_metrics['total_loss']:.6f}",
        "train_gender_acc": f"{train_metrics['gender_acc']:.2f}",
        "train_age_mae": f"{train_metrics['age_mae']:.2f}",
        "train_age_rmse": f"{train_metrics['age_rmse']:.2f}",
        "train_age_bin_acc": f"{train_metrics['age_bin_acc']:.2f}",
        "val_gender_loss": f"{val_metrics['gender_loss']:.6f}",
        "val_age_ce": f"{val_metrics['age_ce']:.6f}",
        "val_age_mae_loss": f"{val_metrics['age_mae_loss']:.6f}",
        "val_age_loss": f"{val_metrics['age_loss']:.6f}",
        "val_total_loss": f"{val_metrics['total_loss']:.6f}",
        "val_gender_acc": f"{val_metrics['gender_acc']:.2f}",
        "val_age_mae": f"{val_metrics['age_mae']:.2f}",
        "val_age_rmse": f"{val_metrics['age_rmse']:.2f}",
        "val_age_bin_acc": f"{val_metrics['age_bin_acc']:.2f}",
        "time": elapsed,
    }

    # 写入 CSV
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)

    # ── 控制台格式化输出 ──
    label_map = {
        "Gender Loss": ("train_gender_loss", "val_gender_loss", ".6f"),
        "Age CE":      ("train_age_ce", "val_age_ce", ".6f"),
        "Age MAE Loss": ("train_age_mae_loss", "val_age_mae_loss", ".6f"),
        "Age Loss":    ("train_age_loss", "val_age_loss", ".6f"),
        "Total Loss":  ("train_total_loss", "val_total_loss", ".6f"),
        "Gender Acc":  ("train_gender_acc", "val_gender_acc", ".2f", "%"),
        "Age MAE":     ("train_age_mae", "val_age_mae", ".2f", " yrs"),
        "Age RMSE":    ("train_age_rmse", "val_age_rmse", ".2f", " yrs"),
        "Age Bin Acc": ("train_age_bin_acc", "val_age_bin_acc", ".2f", "%"),
    }

    lines = [
        f"+{'-'*78}+",
        f"| Epoch {epoch:3d}  |  LR: {lr:.2e}  |  Time: {elapsed}",
        f"+{'-'*39}+{'-'*39}+",
        f"| {'Metric':<20s} {'Train':>17s} | {'Val':>17s} |",
        f"+{'-'*39}+{'-'*39}+",
    ]

    for label, (train_key, val_key, fmt, *suffix) in label_map.items():
        t_val = float(row[train_key])
        v_val = float(row[val_key])
        sfx = suffix[0] if suffix else ""
        lines.append(
            f"| {label:<20s} {t_val:{fmt}}{sfx:>5s} | {v_val:{fmt}}{sfx:>5s} |"
        )

    lines.append(f"+{'-'*78}+")
    print("\n".join(lines))
