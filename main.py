"""
MultiFace — 基于人脸图像的性别与年龄预测系统
================================================
入口脚本：直接运行以启动完整训练流程。

用法:
    python main.py               # 训练模型
    python dataset.py            # 检查数据集
    python model.py              # 检查模型结构

项目结构:
    main.py        - 程序入口，训练主流程
    config.py      - 超参数与路径配置
    dataset.py     - 数据加载、预处理、划分
    model.py       - 多任务学习模型定义
    train.py       - 训练/验证/评估核心函数
"""

import time

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import (
    DEVICE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    LR_STEP_SIZE, LR_GAMMA, EARLY_STOP_PATIENCE,
    MODEL_SAVE_PATH, LOG_PATH,
    print_config,
)
from dataset import create_dataloaders
from model import build_model, get_loss_functions
from train import train_one_epoch, evaluate, log_epoch, format_time


def main():
    """完整的训练/验证/测试主流程"""
    print_config()

    # ── 1. 数据准备 ──
    print("\n" + "=" * 60)
    print("  加载数据集...")
    print("=" * 60)
    train_loader, val_loader, test_loader, stats = create_dataloaders()

    # ── 2. 模型、损失函数、优化器 ──
    print("\n" + "=" * 60)
    print("  构建模型...")
    print("=" * 60)
    model = build_model()
    gender_criterion, age_criterion = get_loss_functions()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    print(f"\n[优化器] AdamW, lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY}")
    print(f"[调度器] StepLR, step_size={LR_STEP_SIZE}, gamma={LR_GAMMA}")

    # ── 3. 训练循环 ──
    print("\n" + "=" * 60)
    print("  开始训练")
    print("=" * 60)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # --- 训练 ---
        train_metrics = train_one_epoch(
            model, train_loader, gender_criterion, age_criterion,
            optimizer, epoch,
        )

        # --- 验证 ---
        val_metrics = evaluate(
            model, val_loader, gender_criterion, age_criterion,
        )

        # --- 学习率调整 ---
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # --- 耗时 ---
        elapsed = format_time(time.time() - epoch_start)

        # --- 日志输出 ---
        log_epoch(epoch, train_metrics, val_metrics, current_lr, elapsed)

        # --- 模型保存 (基于验证集总损失) ---
        val_total = val_metrics["total_loss"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, MODEL_SAVE_PATH)
            print(f"  >>> 模型已保存 (最佳验证 Loss: {best_val_loss:.4f}) <<<")
        else:
            patience_counter += 1

        # --- 早停 ---
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[早停] 验证 loss 已 {EARLY_STOP_PATIENCE} 轮未改善，停止训练。")
            break

    # ── 4. 训练总结 ──
    total_time = format_time(time.time() - train_start)
    print("\n" + "=" * 60)
    print("  训练完成")
    print("=" * 60)
    print(f"  总耗时:        {total_time}")
    print(f"  最佳 epoch:    {best_epoch}")
    print(f"  最佳验证 Loss: {best_val_loss:.4f}")

    # ── 5. 加载最佳模型并测试 ──
    print("\n" + "=" * 60)
    print("  加载最佳模型，在测试集上评估...")
    print("=" * 60)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, gender_criterion, age_criterion)

    print(f"\n  ┌{'─'*36}┐")
    print(f"  │ {'测试集最终结果':^34s} │")
    print(f"  ├{'─'*36}┤")
    print(f"  │ {'性别准确率 (Gender Acc):':<22s} {test_metrics['gender_acc']:>6.2f}%  │")
    print(f"  │ {'年龄 MAE (Age MAE):':<22s} {test_metrics['age_mae']:>6.2f} yrs │")
    print(f"  │ {'年龄 RMSE (Age RMSE):':<22s} {test_metrics['age_rmse']:>6.2f} yrs │")
    print(f"  │ {'年龄分箱准确率 (Bin Acc):':<22s} {test_metrics['age_bin_acc']:>6.2f}%  │")
    print(f"  └{'─'*36}┘")

    print(f"\n  模型已保存到: {MODEL_SAVE_PATH}")
    print(f"  训练日志已保存到: {LOG_PATH}")

    return test_metrics


if __name__ == "__main__":
    main()
