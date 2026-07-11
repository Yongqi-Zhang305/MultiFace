"""
MultiFace — 基于人脸图像的性别与年龄预测系统
================================================
入口脚本：直接运行以启动完整训练流程。

训练策略:
    - 年龄: 101 类 Softmax 分类 + 期望回归 (DEX), 损失 = CE + λ×MAE
    - 分阶段: 前 STAGE_SPLIT 轮冻结性别头, 训练年龄头和骨干;
              之后解冻, 全参数联合训练
    - 差异学习率: 骨干 (低) < 性别头 (中) < 年龄头 (高)

用法:
    python main.py               # 训练模型
    python dataset.py            # 检查数据集
    python model.py              # 检查模型结构
"""

import time
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import (
    DEVICE, EPOCHS, WEIGHT_DECAY,
    BACKBONE_LR, GENDER_HEAD_LR, AGE_HEAD_LR,
    LR_STEP_SIZE, LR_GAMMA, EARLY_STOP_PATIENCE,
    STAGE_SPLIT, MODEL_SAVE_PATH, LOG_PATH,
    print_config,
)
from dataset import create_dataloaders
from model import build_model, get_loss_functions
from train import train_one_epoch, evaluate, log_epoch, format_time


def freeze_module(module, freeze=True):
    """冻结/解冻一个 nn.Module 的所有参数"""
    for param in module.parameters():
        param.requires_grad = not freeze


def build_optimizer(model, stage):
    """根据训练阶段构建优化器，不同的参数组使用不同的学习率。

    Stage 1: 只训练骨干 + 年龄头 (性别头冻结)
    Stage 2: 训练全部, 但三个组件各用不同的 LR
    """
    if stage == 1:
        # 冻结性别头 → requires_grad=False, 不会出现在 optimizer 中
        param_groups = [
            {"params": model.backbone.parameters(),  "lr": BACKBONE_LR},
            {"params": model.age_head.parameters(),   "lr": AGE_HEAD_LR},
        ]
    else:
        param_groups = [
            {"params": model.backbone.parameters(),   "lr": BACKBONE_LR},
            {"params": model.gender_head.parameters(), "lr": GENDER_HEAD_LR},
            {"params": model.age_head.parameters(),    "lr": AGE_HEAD_LR},
        ]

    return optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


def main():
    """分阶段训练主流程"""
    print_config()

    # ── 1. 数据准备 ──
    print("\n" + "=" * 60)
    print("  加载数据集...")
    print("=" * 60)
    train_loader, val_loader, test_loader, stats = create_dataloaders()

    # ── 2. 模型构建 ──
    print("\n" + "=" * 60)
    print("  构建模型...")
    print("=" * 60)
    model = build_model()
    gender_criterion = get_loss_functions()

    # ── 3. 阶段一: 冻结性别头, 只训练年龄头+骨干 ──
    stage1_epochs = int(EPOCHS * STAGE_SPLIT)
    stage2_epochs = EPOCHS - stage1_epochs

    print(f"\n{'='*60}")
    print(f"  阶段一 (Epoch 1-{stage1_epochs}): 冻结性别头, 训练年龄头+骨干")
    print(f"  性别头 LR: 0 (冻结) | 骨干 LR: {BACKBONE_LR} | 年龄头 LR: {AGE_HEAD_LR}")
    print(f"{'='*60}")

    freeze_module(model.gender_head, freeze=True)
    optimizer = build_optimizer(model, stage=1)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    print(f"\n[优化器] 参数组数: {len(optimizer.param_groups)}")
    for i, pg in enumerate(optimizer.param_groups):
        print(f"  组{i}: {sum(p.numel() for p in pg['params']):,} params, lr={pg['lr']}")

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_start = time.time()

    # ── 阶段一训练 ──
    for epoch in range(1, stage1_epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, gender_criterion, optimizer, epoch,
        )
        val_metrics = evaluate(model, val_loader, gender_criterion)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = format_time(time.time() - epoch_start)

        log_epoch(epoch, train_metrics, val_metrics, current_lr, elapsed)

        val_total = val_metrics["total_loss"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }, MODEL_SAVE_PATH)
            print(f"  >>> 模型已保存 (最佳验证 Loss: {best_val_loss:.4f}) <<<")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n[早停] 验证 loss 已 {EARLY_STOP_PATIENCE} 轮未改善，停止训练。")
            break

    # ── 4. 阶段二: 解冻性别头, 全参数联合训练 ──
    if stage2_epochs > 0 and patience_counter < EARLY_STOP_PATIENCE:
        print(f"\n{'='*60}")
        print(f"  阶段二 (Epoch {stage1_epochs+1}-{EPOCHS}): 解冻性别头, 联合训练")
        print(f"  性别头 LR: {GENDER_HEAD_LR} | 骨干 LR: {BACKBONE_LR} | 年龄头 LR: {AGE_HEAD_LR}")
        print(f"{'='*60}")

        freeze_module(model.gender_head, freeze=False)
        # 重新构建优化器以包含性别头参数组
        optimizer = build_optimizer(model, stage=2)
        scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

        # 重置 patience 计数器 (阶段切换后给模型更多适应时间)
        patience_counter = 0

        print(f"\n[优化器] 参数组数: {len(optimizer.param_groups)}")
        for i, pg in enumerate(optimizer.param_groups):
            print(f"  组{i}: {sum(p.numel() for p in pg['params']):,} params, lr={pg['lr']}")

        for epoch in range(stage1_epochs + 1, EPOCHS + 1):
            epoch_start = time.time()

            train_metrics = train_one_epoch(
                model, train_loader, gender_criterion, optimizer, epoch,
            )
            val_metrics = evaluate(model, val_loader, gender_criterion)

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            elapsed = format_time(time.time() - epoch_start)

            log_epoch(epoch, train_metrics, val_metrics, current_lr, elapsed)

            val_total = val_metrics["total_loss"]
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_epoch = epoch
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                }, MODEL_SAVE_PATH)
                print(f"  >>> 模型已保存 (最佳验证 Loss: {best_val_loss:.4f}) <<<")
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n[早停] 验证 loss 已 {EARLY_STOP_PATIENCE} 轮未改善，停止训练。")
                break

    # ── 5. 训练总结 ──
    total_time = format_time(time.time() - train_start)
    print("\n" + "=" * 60)
    print("  训练完成")
    print("=" * 60)
    print(f"  总耗时:        {total_time}")
    print(f"  最佳 epoch:    {best_epoch}")
    print(f"  最佳验证 Loss: {best_val_loss:.4f}")

    # ── 6. 加载最佳模型并测试 ──
    print("\n" + "=" * 60)
    print("  加载最佳模型，在测试集上评估...")
    print("=" * 60)

    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, gender_criterion)

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
