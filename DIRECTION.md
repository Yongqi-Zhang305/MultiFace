# MultiFace 项目完全代码讲解

> 面向零基础读者的深度学习项目逐行讲解。每一节先讲清楚「概念是什么」，再结合本项目代码说明「怎么用的」。

---

## 目录

- [第〇章：前置概念 —— 把这些搞懂了，代码才能看懂](#第〇章前置概念--把这些搞懂了代码才能看懂)
  - [0.1 什么是机器学习](#01-什么是机器学习)
  - [0.2 什么是神经网络](#02-什么是神经网络)
  - [0.3 什么是卷积神经网络 CNN](#03-什么是卷积神经网络-cnn)
  - [0.4 什么是 ResNet](#04-什么是-resnet)
  - [0.5 什么是迁移学习 / 预训练模型](#05-什么是迁移学习--预训练模型)
  - [0.6 什么是分类与回归](#06-什么是分类与回归)
  - [0.7 什么是多任务学习](#07-什么是多任务学习)
  - [0.8 什么是损失函数](#08-什么是损失函数)
  - [0.9 什么是优化器](#09-什么是优化器)
  - [0.10 什么是反向传播](#010-什么是反向传播)
  - [0.11 什么是 Epoch、Batch、Iteration](#011-什么是-epochbatchiteration)
  - [0.12 什么是过拟合、Dropout、数据增强](#012-什么是过拟合dropout数据增强)
  - [0.13 什么是训练集 / 验证集 / 测试集](#013-什么是训练集--验证集--测试集)
  - [0.14 什么是学习率调度与早停](#014-什么是学习率调度与早停)
- [第一章：项目总览](#第一章项目总览)
- [第二章：config.py 逐行讲解](#第二章configpy-逐行讲解)
- [第三章：dataset.py 逐行讲解](#第三章datasetpy-逐行讲解)
- [第四章：model.py 逐行讲解](#第四章modelpy-逐行讲解)
- [第五章：train.py 逐行讲解](#第五章trainpy-逐行讲解)
- [第六章：main.py 逐行讲解](#第六章mainpy-逐行讲解)
- [第七章：运行与理解输出](#第七章运行与理解输出)

---

## 第〇章：前置概念 —— 把这些搞懂了，代码才能看懂

### 0.1 什么是机器学习

**通俗理解：**

传统的计算机程序是「人写规则，计算机执行」。比如你想判断一张照片里的人是男是女，传统方法需要你手动写规则：「如果有长头发 → 可能是女」「如果有胡子 → 可能是男」。但问题是：不是所有女生都长头发，不是所有男生都有胡子，规则永远写不完。

机器学习的思路反过来了：你不再写规则，而是给计算机看**大量的例子**（照片 + 标签「男/女」），让计算机**自己从例子中找到规律**。这个过程叫做**训练**。

**三个核心要素：**
1. **数据**：输入（人脸照片）+ 标签（性别、年龄）
2. **模型**：一个数学函数，输入照片的像素值，输出预测结果
3. **训练**：不断调整这个数学函数的参数，让它预测得越来越准

**类比：** 就像教小孩认识动物。你不是给小孩写一本《动物识别规则手册》，而是给他看很多动物图片，告诉他「这是猫」「这是狗」。看得多了，小孩自己就学会了。

---

### 0.2 什么是神经网络

**通俗理解：**

神经网络是模拟人脑神经元工作方式的一种数学模型。

想象一个工厂流水线：
```
[原材料] → [工位1加工] → [工位2加工] → [工位3加工] → [成品]
```

神经网络的每一层就是一个「工位」。输入数据经过一层一层的加工，最终变成输出结果。

**数学层面：**

一个最简单的神经网络层做的事情就是：
```
输出 = 激活函数(输入 × 权重 + 偏置)
```

- **输入**：给这一层的原始数据
- **权重（Weight）**：每个输入的重要性，是模型要学习的参数
- **偏置（Bias）**：一个额外的调整项
- **激活函数**：一个非线性函数，让网络能学习复杂模式

整个训练的过程，就是不断**调整所有权重和偏置**，让输出越来越接近正确答案。

**在本项目中：** 我们用的神经网络叫 ResNet18，它是一个已经设计好的、专门处理图像的神经网络结构。我们会在 0.4 节详细介绍它。

---

### 0.3 什么是卷积神经网络 CNN

**通俗理解：**

普通神经网络处理图像时有一个大问题：它把图片的每个像素都当作独立的输入。但一张 224×224 的彩色图片有 224 × 224 × 3 = 150,528 个像素值，如果每个像素都连到下一层的每个神经元，参数数量会爆炸，而且完全丢失了「相邻像素之间的关系」。

卷积神经网络（CNN）解决了这个问题。它的核心思想是：**用一个小窗口在图片上滑动，每次只看一小块区域**。

**类比：** 你阅读这篇文章时，不是同时看全部文字，而是一个词一个词地扫过去。每个词只是全文的一小部分，但你能通过局部信息理解整体含义。CNN 也是这样——用一个叫「卷积核」的小方块在图片上滑动，每次只「看」图片的一小块区域。

**CNN 的核心组件：**
1. **卷积层（Convolutional Layer）**：滑动窗口，提取局部特征（边缘、纹理、形状等）
2. **池化层（Pooling Layer）**：缩小图片尺寸，减少计算量，保留最重要的信息
3. **全连接层（Fully Connected Layer / FC）**：最后的决策层，把提取到的特征综合起来做判断

**在本项目中：** ResNet18 就是一个成熟的 CNN 架构，它由几十层卷积层 + 池化层堆叠而成。我们直接用它作为「特征提取器」。

---

### 0.4 什么是 ResNet

**全称：** Residual Network（残差网络）

**为什么需要 ResNet？**

直觉上，网络越深（层数越多），能力应该越强。但实践中发现，当网络深到一定程度后，反而效果变差——不是过拟合，而是**训练不动了**。这叫做「梯度消失」问题。

打个比方：你在一个很长的传话游戏中，第一个人说「明天吃饺子」，经过 50 个人口口相传，最后变成了「火星上有兔子」。信息在传递过程中慢慢丢失了。

**ResNet 的解决方案 —— 残差连接（Skip Connection）：**

ResNet 的巧妙之处是加了一条「快速通道」：
```
正常路径：输入 → 卷积层1 → 卷积层2 → 输出
残差连接：输入 ─────────────────────→ + → 输出
```

也就是说，输出 = 卷积处理后的结果 + 原始输入。这样一来，即使卷积层什么都没学到，至少原始信息还能原封不动传过去。这条「快速通道」确保了信息不会在深层网络中丢失。

**常见的 ResNet 变体：**
- ResNet18：18 层（本项目使用的）
- ResNet34：34 层
- ResNet50：50 层

数字越大，网络越深，表达能力越强，但计算量也越大。

**在本项目中：** 我们在 `config.py` 中设置 `BACKBONE = "resnet18"`，在 `model.py` 中加载它。ResNet18 充当整个模型的「骨干」，负责从人脸图片中提取有用的特征。

---

### 0.5 什么是迁移学习 / 预训练模型

**通俗理解：**

你学会骑自行车之后，学骑摩托车会快很多，因为「保持平衡」和「掌握方向」的基础技能是相通的。你不需要从头学起。

迁移学习就是这个道理。ImageNet 是一个包含 120 万张图片、1000 个类别的大规模数据集。在 ImageNet 上训练好的 ResNet 已经学会了识别通用的视觉特征（边缘、形状、纹理、物体部件等）。

我们「借用」这个已经在 ImageNet 上训练好的 ResNet，只需要把最后几层替换成我们自己的任务（性别分类 + 年龄预测），然后在 UTKFace 人脸数据上稍作微调即可。

**类比：** 一个物理系博士转行做金融。他不需要重新从小学读起——他的数学功底已经很好，只需要学一些金融专业知识就能胜任。

**和「从头训练」的区别：**
- **从头训练（from scratch）**：所有参数随机初始化，需要海量数据和计算资源
- **迁移学习（transfer learning）**：用预训练权重初始化，只需要少量数据就能达到好效果

**在本项目中：** 我们在 `config.py` 中设置 `PRETRAINED = True`，在 `model.py` 中通过 `weights="IMAGENET1K_V1"` 加载预训练权重。

---

### 0.6 什么是分类与回归

这是深度学习中两种最基本的问题类型：

| | 分类（Classification） | 回归（Regression） |
|---|---|---|
| **输出** | 离散的类别 | 连续的数值 |
| **例子** | 男/女、猫/狗/鸟 | 年龄 23.5 岁、房价 350 万 |
| **评价指标** | 准确率（Accuracy） | 平均绝对误差（MAE） |
| **最终层激活函数** | Softmax | 无（直接输出数值） |

**本项目中有两个任务：**
- **性别预测**：分类问题。答案只有「男」或「女」两种。输出是 2 个分数，取分数高的那个作为答案。
- **年龄预测**：回归问题。答案是连续的数值（如 25.3 岁）。输出是一个数值，越接近真实年龄越好。

---

### 0.7 什么是多任务学习

**通俗理解：**

多任务学习就是让一个模型同时学习多个相关任务。

比如你同时学弹吉他和弹钢琴——虽然它们是不同的乐器，但乐理、节奏感、手指灵活性是共通的。一起学反而比分别学效果更好，因为共通的能力可以在两个任务之间共享。

**本项目中的体现：**

我们只有一个 ResNet18 骨干网络，但它上面有两个「头」（Head）：
```
                          ┌→ 性别分类头 → 输出: 男/女
ResNet18（共享特征提取）→
                          └→ 年龄分类头 → 输出: 117类概率 → 期望年龄
```

- 「男性」和「老年人」可能有相关的视觉特征（如皱纹、发际线）
- 共享的骨干网络能同时学到对两个任务都有用的特征
- 相比训练两个独立模型，多任务学习更高效，泛化能力也更好

---

### 0.8 什么是损失函数

**通俗理解：**

损失函数（Loss Function）是衡量「模型预测得有多差」的一个数字。

就像考试：你每做错一道题，就扣几分。损失函数就是批卷老师，告诉你「你这次考得有多差」。训练的目标就是让这个「扣分」越来越小。

**类比：**

你在练习投篮。每次投篮后，你看到的不是「进没进」，而是一个具体的分数——偏离篮筐多少厘米。损失函数就是那个量尺。你的目标是把偏离距离（损失）降到最小。

**本项目中的损失函数：**

| 任务 | 损失函数 | 全称 | 含义 |
|------|---------|------|------|
| 性别 | CrossEntropyLoss | 交叉熵损失 | 衡量两个概率分布之间的差异 |
| 年龄 | CE + λ×MAE | 联合损失 | CE: 117类分类的交叉熵; MAE: 期望年龄与真实年龄的L1距离 |

**为什么年龄用联合损失（CE + λ×MAE）？**

年龄既是一个「类别问题」（几岁），又天然带有「距离信息」（30岁和31岁接近，和60岁很远）。单纯用回归（L1Loss）难以训练，单纯用分类（CrossEntropy）丢失距离。联合损失取了两者之长：

- **CE（交叉熵）**：让模型输出 117 个类别的概率分布，分类任务稳定好训
- **MAE（期望年龄 vs 真实年龄）**：从概率分布算期望年龄 `sum(p_i × i)`，用 L1 约束它靠近真实年龄，保留了「30 比 60 更接近 31」的距离信息
- **λ = 0.5**：调节两项的权重，CE 主导训练方向，MAE 提供距离感知

UTKFace 数据集中有 0~116 岁的人，直接用 L1Loss 回归的话大龄误差会主导梯度，而联合损失通过分类把误差分散到 117 个类别上，训练更稳定。

---

### 0.9 什么是优化器

**通俗理解：**

损失函数告诉你「错得有多离谱」，优化器（Optimizer）则负责「该怎么改」。

继续投篮的比喻：损失函数告诉你「偏离篮筐 30cm」，优化器就是你的大脑，决定「下一次手腕多用点力，角度稍微调整一下」。

**数学层面：**

优化器的核心任务是**调整模型的参数（权重和偏置）**，使得下一次预测的损失更小。

**本项目使用的优化器 —— **AdamW**，并且不同组件使用**差异化学习率**：**

AdamW 是目前最主流的优化器之一。它的核心思想是**自适应学习率**：每个参数有不同的调整速度，有些参数需要大步调整，有些需要小步微调。

- **`lr`（Learning Rate，学习率）**：控制每次调整的幅度。太大会跳过头，太小会学得太慢。
- **`weight_decay`（权重衰减）**：一种防止过拟合的技术，给参数施加一个「变小」的压力。

**类比：** 学习率就像你投篮时每次调整的幅度。调太大了，你可能从偏左直接跳到偏右；调太小了，你投了 100 个球还偏着一个方向。

---

### 0.10 什么是反向传播

**通俗理解：**

反向传播（Backpropagation）是神经网络的学习机制。它回答了「每个参数对最终错误的贡献是多少」这个问题。

**正向过程：**
```
输入图片 → 第1层计算 → 第2层计算 → ... → 输出结果 → 计算损失（和正确答案比）
```

**反向过程：**
```
损失 → 反向逐层传递误差 → 算出每个参数该调多少 → 优化器更新参数
```

**类比：** 多米诺骨牌。正向传播是一路推倒骨牌，反向传播是倒着回溯，找出是哪一块骨牌的摆放角度导致了最终倒下的方向不对。

**本项目中：** PyTorch 的 `loss.backward()` 就是执行反向传播，`optimizer.step()` 就是根据反向传播的结果更新参数。你不需要手动计算任何梯度，PyTorch 全自动。

---

### 0.11 什么是 Epoch、Batch、Iteration

这三个概念容易混淆，用食堂打饭的比喻一次讲清楚：

- **Epoch（轮次）**：把整个数据集从头到尾过一遍。好比全校 1000 个学生都来食堂吃了一遍饭。
- **Batch（批次）**：每次处理的一小批数据。好比食堂一个窗口一次只能服务 64 个学生。
- **Iteration（迭代）**：处理完一个 batch 就算一次迭代。1000 个学生，每批 64 人，需要 1000÷64 ≈ 16 次迭代才能完成 1 个 Epoch。

**为什么要分批？**

如果把全部 23708 张图片一次性喂给模型，你的显卡内存会爆炸。分批处理是最基本的工程手段。

**在本项目中：**
- `BATCH_SIZE = 128`（每批 128 张图片）
- `EPOCHS = 30`（整个数据集过 30 遍）
- 16593 张训练图片 ÷ 128 ≈ 130 次迭代 = 1 个 epoch

---

### 0.12 什么是过拟合、Dropout、数据增强

**过拟合（Overfitting）：**

你把数学课本上的 10 道例题背得滚瓜烂熟，考试时题目稍微变一下就不会做了。这就是过拟合——模型「记住」了训练数据，而不是「理解」了内在规律。

**Dropout（随机失活）：**

训练时，每个 batch 随机「关掉」一部分神经元（让它们暂时不工作）。这迫使剩下的神经元不能依赖某个特定的「同事」，必须自己学会鲁棒的特征。

**类比：** 一个团队做项目，每次随机让 40% 的成员休假。这样一来，项目文档必须写得足够清楚，确保任何成员休假后其他人也能顶上。团队整体能力反而更强了。

**本项目中：** `nn.Dropout(p=0.4)` 表示每次随机丢弃 40% 的神经元。

**数据增强（Data Augmentation）：**

把现有数据做一些随机变换，创造出「新」数据。比如把人脸图片随机翻转、旋转、调亮调暗。这样模型看到的就不再是同一批数据反复出现，而是每次都有微小变化。

**本项目中：** `get_transforms(is_train=True)` 函数里包含了：
- 随机水平翻转
- 随机旋转（±15°）
- 颜色抖动（亮度、对比度、饱和度随机变化）
- 随机仿射变换（缩放、平移）

---

### 0.13 什么是训练集 / 验证集 / 测试集

把数据集分成三份，各司其职：

| 集合 | 占比 | 用途 | 类比 |
|------|------|------|------|
| **训练集** | 70% | 用来训练模型，调整参数 | 平时的作业和练习题 |
| **验证集** | 15% | 训练过程中评估模型，选最佳参数 | 模拟考试，检测学习效果 |
| **测试集** | 15% | 训练完全结束后，最终评估 | 期末考试，见真章 |

**关键原则：测试集在整个训练过程中绝对不能碰！** 它只能用于最后的最终评估。如果你根据测试集的表现回去调整模型，那测试集就变成了验证集，失去了「最终考试」的意义。

**在本项目中：** `TRAIN_RATIO = 0.70`, `VAL_RATIO = 0.15`, `TEST_RATIO = 0.15`。

---

### 0.14 什么是学习率调度与早停

**学习率调度（Learning Rate Scheduling）：**

训练前期，学习率可以大一点，快速接近目标；训练后期，学习率要小一点，精细调整。

**类比：** 你开车去一个陌生的地方。一开始在高速上，可以开快一点（高学习率）；快到了的时候，减到低速在小巷里慢慢找精确位置（低学习率）。

**本项目中：** StepLR 调度器每 15 个 epoch 把学习率乘以 0.5。比如：
- Epoch 1-15：lr = 0.0001
- Epoch 16-30：lr = 0.00005

**早停（Early Stopping）：**

如果验证集的损失连续 N 个 epoch 都不再下降，说明模型已经学得差不多了，继续训练只会浪费时间甚至过拟合。

**本项目中：** `EARLY_STOP_PATIENCE = 10`，连续 10 轮验证损失不下降就自动停止。

---

## 第一章：项目总览

在理解了以上所有概念之后，我们来看这个项目的全貌。

### 1.1 这个项目做什么？

输入一张人脸照片 → 模型同时输出：
- **性别**：男 或 女
- **年龄**：一个数值（如 28.5 岁）

### 1.2 五个文件的分工

| 文件 | 职责 | 类比 |
|------|------|------|
| `config.py` | 存放所有可调整的参数 | 菜谱 —— 决定放多少盐、多少油 |
| `dataset.py` | 加载图片、解析标签、数据增强 | 备菜 —— 买菜、洗菜、切菜 |
| `model.py` | 定义神经网络的结构 | 灶台 —— 搭好锅碗瓢盆 |
| `train.py` | 训练一个 epoch、验证、打日志 | 烹饪动作 —— 翻炒、尝味、记录 |
| `main.py` | 串联整个训练流程 | 主厨 —— 指挥全局，什么先做什么后做 |

### 1.3 数据流全景图

```
config.py（参数配置）
    │
    ▼
dataset.py（加载图片 → 数据增强 → 分成训练/验证/测试三份）
    │
    ▼
model.py（构建 ResNet18 骨干 + 性别头 + 年龄头）
    │
    ▼
main.py（主循环：for epoch in 1..30:）
    ├── train.py → train_one_epoch()   # 训练一轮
    ├── train.py → evaluate()          # 验证一轮
    └── train.py → log_epoch()         # 打印指标
    │
    ▼
output/best_model.pth  # 保存最佳模型
output/training_log.csv  # 保存训练日志
```

---

## 第二章：config.py 逐行讲解

`config.py` 是整个项目的「控制面板」。所有可以调整的设置都集中在这里。
如果你想把 ResNet18 换成 ResNet50，改这一个文件就够了。

### 2.1 文件头部的 import

```python
import os
import torch
```
- `os`：操作系统接口，用来处理文件路径（比如拼接目录路径）
- `torch`：PyTorch 深度学习框架的核心库

### 2.2 路径配置（第 8-19 行）

```python
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
```

这行代码的作用是**自动找到项目根目录**，无论你把项目复制到哪个盘、哪台电脑上都能用。逐层拆解：
- `__file__`：当前文件（config.py）的路径
- `os.path.abspath(__file__)`：转成绝对路径，比如 `D:\MultiFace\config.py`
- `os.path.dirname(...)`：取父目录，得到 `D:\MultiFace`

```python
DATA_DIR = os.path.join(ROOT_DIR, "data", "UTKFace")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LOG_PATH = os.path.join(OUTPUT_DIR, "training_log.csv")
```
- `os.path.join()` 是跨平台的路径拼接方法。在 Windows 上它会用 `\`，在 Linux/Mac 上会用 `/`
- `DATA_DIR`：数据集图片存放的文件夹
- `OUTPUT_DIR`：训练产出的文件夹（模型文件 + 日志）
- `MODEL_SAVE_PATH`：训练过程中保存最佳模型的位置（`.pth` 是 PyTorch 模型文件的后缀）
- `LOG_PATH`：CSV 格式的训练日志文件

```python
os.makedirs(OUTPUT_DIR, exist_ok=True)
```
- 如果 `output/` 文件夹不存在，就创建它
- `exist_ok=True` 表示如果文件夹已经存在，不报错

### 2.3 数据集配置（第 21-29 行）

```python
IMG_SIZE = 224
```
输入图片的宽高，单位是像素。224×224 是 ResNet 系列模型的标准输入尺寸。

```python
BATCH_SIZE = 128
```
每次喂给模型的图片数量。128 是一个比较均衡的选择——既保证 GPU 利用率，又不会爆显存。你可以根据自己的显卡调整：显存大就调大，显存小就调小。

```python
NUM_WORKERS = 4
```
DataLoader 使用多少个并行进程来加载图片。
- 设为 0：主进程自己加载（慢，但兼容性最好）
- 设为 4：4 个子进程并行加载（快，Windows 上需要特殊处理）

```python
TRAIN_RATIO = 0.70   # 70% 数据用于训练
VAL_RATIO = 0.15     # 15% 用于验证
TEST_RATIO = 0.15    # 15% 用于测试
```

```python
AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 120]
```
年龄分箱的边界。用于计算「年龄分箱准确率」：比如预测 23 岁，真实 28 岁，都落在 [20, 30) 这个区间里，就算预测正确。这种评估方式比精确 MAE 更宽松，也更有实际意义（知道一个人是「二十多岁」vs「六十多岁」往往比精确到个位数更重要）。

### 2.4 模型配置（第 34-40 行）

```python
BACKBONE = "resnet18"
```
选择哪个预训练网络作为骨干。可选：
- `"resnet18"`：18 层，最快，本项目默认
- `"resnet34"`：34 层，稍慢，稍强
- `"resnet50"`：50 层，更慢，更强
- `"mobilenet_v2"`：为移动端设计的轻量网络，最快但精度略低

```python
PRETRAINED = True
```
是否使用 ImageNet 预训练权重。设为 `True` 就是迁移学习（见 0.5 节），设为 `False` 就是从头训练。

```python
GENDER_CLASSES = 2
```
性别分类的类别数。2 就是「男」和「女」。

```python
NUM_AGE_CLASSES = 117
```
年龄类别总数。0-116 岁每个年龄一个类别，完全覆盖 UTKFace 数据集的年龄范围。为什么不是 101？因为 UTKFace 中年龄最大到 116 岁，用 117 类才能确保每个样本都有精准对应的类别，不会被钳位（clamp）到 100 导致失真。

```python
AGE_CE_LAMBDA = 0.5
```
联合损失中 MAE 的权重。年龄总损失 = CE（117 类交叉熵）+ 0.5 × MAE（期望年龄的 L1 距离）。
- 之前用纯回归时 AGE_LOSS_WEIGHT = 0.4，现在因为 CE 已经承担了主要训练信号，MAE 只作为辅助，lambda = 0.5 是一个较平衡的值
- 如果权重是 1.0，年龄损失会主导训练，性别任务学不好
- 0.4 是通过实验找出来的平衡点

### 2.5 训练配置

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```
自动检测有没有 NVIDIA 显卡。有就用 GPU（cuda），没有就用 CPU。

```python
EPOCHS = 30
```
最多训练 30 轮。实际可能因为早停在 15-20 轮左右提前结束。

```python
WEIGHT_DECAY = 1e-4
```
权重衰减，一种正则化手段，防止模型参数过大导致过拟合。

```python
STAGE_SPLIT = 0.5
```
分阶段训练的前后比例。0.5 表示前 50%（15 轮）为阶段一（冻结性别头），后 50%（15 轮）为阶段二（全参数联合训练）。

```python
BACKBONE_LR = 1e-5
GENDER_HEAD_LR = 1e-4
AGE_HEAD_LR = 2e-4
```
**不同组件使用不同的学习率**，这是本项目的一个重要训练策略：
- 骨干网络（ResNet18）：`1e-5`，最低。ImageNet 预训练权重已经很好，只需要小幅微调
- 性别头：`1e-4`，中等。从头训练但任务简单（二分类）
- 年龄头：`2e-4`，最高。从头训练且任务最复杂（117 类分类+回归）

```python
LR_STEP_SIZE = 15
LR_GAMMA = 0.5
```
学习率调度参数：每 15 个 epoch，**所有**学习率乘以 0.5。比如：
- Epoch 1-15：backbone=1e-5, gender=1e-4, age=2e-4
- Epoch 16+：backbone=5e-6, gender=5e-5, age=1e-4

```python
EARLY_STOP_PATIENCE = 10
```
早停的耐心值。连续 10 个 epoch 验证损失没有创新低，就停止训练。

### 2.6 print_config() 函数

```python
def print_config():
    print("=" * 60)
    print("   MultiFace — 人脸性别与年龄预测项目")
    ...
```
`"=" * 60` 的意思是打印 60 个等号，形成一条分割线。这个函数纯粹是为了在训练开始前打印一份配置摘要，让你确认一切设置正确。

---

## 第三章：dataset.py 逐行讲解

`dataset.py` 负责所有和「数据」相关的工作：读图片、解析标签、数据增强、划分数据集。

### 3.1 导入的库

```python
import os
import numpy as np
from PIL import Image
```
- `os`：处理文件路径
- `numpy`（简写 `np`）：Python 科学计算的基础库，处理数组和数学运算
- `PIL.Image`：Python Imaging Library，用来打开和读取图片文件

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
```
- `torch.utils.data.Dataset`：PyTorch 的数据集基类。我们要继承它来定义自己的数据集
- `torch.utils.data.DataLoader`：把 Dataset 包装成可迭代的批量数据加载器
- `torch.utils.data.random_split`：随机划分数据集
- `torchvision.transforms`：图像变换工具包（缩放、翻转、归一化等）

### 3.2 SubsetDataset 类（第 23-44 行）

```python
class SubsetDataset(Dataset):
```

**类（class）是什么？** 类是一种代码组织方式，把相关的数据（属性）和功能（方法）打包在一起。你可以把它想象成一张「图纸」，而 `SubsetDataset(...)` 就是按照图纸制造出来的一个具体的「产品」（实例）。

`(Dataset)` 表示这个类**继承**了 PyTorch 的 `Dataset` 类——它自动获得 Dataset 的所有基础功能，只需要实现三个关键方法：`__init__`、`__len__`、`__getitem__`。

```python
def __init__(self, base_dataset, indices, transform):
    self.base = base_dataset
    self.indices = indices
    self.transform = transform
```
- `__init__` 是**构造函数**，在创建 SubsetDataset 实例时自动调用
- `base_dataset`：原始的完整数据集（一个 UTKFaceDataset 实例）
- `indices`：一个索引列表，比如 `[0, 3, 5, 7, ...]`，表示「我们要原始数据集的哪几张图」
- `transform`：图像变换函数（数据增强或基础变换）

```python
def __len__(self):
    return len(self.indices)
```
返回这个子集有多少张图片。PyTorch 的 DataLoader 需要知道数据总量来规划每个 epoch。

```python
def __getitem__(self, idx):
    original_idx = self.indices[idx]
    path = self.base.file_paths[original_idx]
    image = Image.open(path).convert("RGB")
    if self.transform:
        image = self.transform(image)
    age = torch.tensor(self.base.ages[original_idx], dtype=torch.float32)
    gender = torch.tensor(self.base.genders[original_idx], dtype=torch.long)
    return image, gender, age
```
这是最核心的方法：给定一个索引 `idx`，返回对应的 `(图片, 性别标签, 年龄标签)`。

逐行解释：
1. `original_idx = self.indices[idx]`：把子集的索引转为原始数据集的索引
2. `path = self.base.file_paths[original_idx]`：从原始数据集拿到图片路径
3. `Image.open(path).convert("RGB")`：用 PIL 打开图片，并确保它是 RGB 三通道彩色图（有些图片可能是灰度图或 RGBA）
4. 如果有 transform，就对图片应用变换（缩放、翻转等）
5. `torch.tensor(..., dtype=torch.float32)`：把年龄值转成 PyTorch 的浮点数张量（回归任务需要浮点数）
6. `torch.tensor(..., dtype=torch.long)`：把性别值转成 PyTorch 的整数张量（分类任务需要整数标签）

**为什么要把数据转成 torch.tensor？** PyTorch 的所有计算都基于张量（tensor）。可以把它理解为 PyTorch 版本的数组，但可以在 GPU 上运行。

### 3.3 UTKFaceDataset 类（第 47-122 行）

这是整个数据加载的核心。我们来分段看。

```python
class UTKFaceDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
```

构造函数接收两个参数：
- `file_paths`：所有图片文件的路径列表
- `transform`：图像变换（可选，默认为 None）

```python
        self.ages = []
        self.genders = []
        self.races = []

        for path in file_paths:
            filename = os.path.basename(path)
            parts = filename.split("_")
```

- 创建三个空列表，用来存放每张图片的标签
- `os.path.basename(path)`：从完整路径中提取文件名。比如 `D:\data\UTKFace\25_1_0_xxx.jpg` → `25_1_0_xxx.jpg`
- `filename.split("_")`：用下划线分割文件名。比如 `"25_1_0_xxx.jpg"` → `["25", "1", "0", "xxx.jpg"]`

**文件名解析规则：** UTKFace 文件名的格式是 `[age]_[gender]_[race]_[timestamp].jpg.chip.jpg`
- `parts[0]` = 年龄
- `parts[1]` = 性别（0=男, 1=女）
- `parts[2]` = 种族

```python
            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2]) if len(parts) >= 3 else -1
            except ValueError:
                age, gender, race = -1, -1, -1
```

`try...except ValueError` 是异常处理：如果文件名字段解析失败（比如某些文件命名不规范），就把这三个标签都设为 -1 作为无效标记，而不是让整个程序崩溃。

```python
        valid_indices = [
            i for i, (a, g) in enumerate(zip(self.ages, self.genders))
            if a >= 0 and g in (0, 1)
        ]
```

这是**列表推导式（list comprehension）**，Python 的一种优雅语法。它的效果是：遍历所有解析结果，筛选出年龄 ≥ 0 且性别是 0 或 1 的合法数据，保留它们的索引。任何被标为 -1 的无效数据都被丢弃。

```python
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        age = torch.tensor(self.ages[idx], dtype=torch.float32)
        gender = torch.tensor(self.genders[idx], dtype=torch.long)
        return image, gender, age
```

和 SubsetDataset 的 `__getitem__` 逻辑相同：打开图片 → 应用变换 → 返回(图片, 性别, 年龄)三元组。

```python
    def get_stats(self):
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
```

`get_stats()` 不是 PyTorch 要求的方法，是我们自己加的辅助函数。它统计数据集的基本信息（男女比例、年龄分布等），在训练开始时打印出来，让你对数据有个直观了解。

### 3.4 get_transforms() 函数（第 125-156 行）

```python
def get_transforms(is_train=True):
```

这个函数返回一个图像变换管道（pipeline）。**训练时**和**测试时**的变换是不同的：

**训练时（is_train=True）：**
```python
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),     # 缩放到 224×224
    transforms.RandomHorizontalFlip(p=0.5),      # 50% 概率左右翻转
    transforms.RandomRotation(degrees=15),        # 随机旋转 ±15°
    transforms.ColorJitter(                       # 颜色随机抖动
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05
    ),
    transforms.RandomAffine(                      # 随机仿射变换
        degrees=0, translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),
    transforms.ToTensor(),                        # 转成 PyTorch 张量
    transforms.Normalize(                         # 归一化
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```
- `transforms.Compose([...])`：把多个变换串联起来，按顺序执行
- 前五个变换都是**数据增强**（见 0.12 节），只在训练时做。它们人为增加数据多样性，防止过拟合
- `ToTensor()`：把 PIL 图片（像素值 0-255）转成 PyTorch 张量（像素值 0.0-1.0）
- `Normalize(mean, std)`：把像素值标准化到 ImageNet 的统计分布。因为我们的 ResNet 是在 ImageNet 上预训练的，用同样的归一化参数能让预训练权重发挥最大效用

**测试时（is_train=False）：**
```python
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 只缩放
    transforms.ToTensor(),                     # 转张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```
测试时不做任何数据增强——我们要评估模型的真实能力，不能让随机变换干扰评估结果。

### 3.5 create_dataloaders() 函数（第 159-216 行）

这是 `dataset.py` 的「总指挥」函数，被 `main.py` 调用。

```python
def create_dataloaders():
    all_files = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".jpeg"):
            all_files.append(os.path.join(DATA_DIR, fname))
```
- `os.listdir(DATA_DIR)`：列出数据文件夹里的所有文件名
- 筛选出 `.jpg`、`.png`、`.jpeg` 结尾的图片文件
- `os.path.join(DATA_DIR, fname)`：拼接成完整的文件路径

```python
    full_dataset = UTKFaceDataset(all_files, transform=None)
    stats = full_dataset.get_stats()
```
创建一个完整的数据集实例（暂时不设 transform），获取统计信息。

```python
    total = len(full_dataset)
    train_size = int(total * TRAIN_RATIO)   # 70% = 16593 张
    val_size = int(total * VAL_RATIO)        # 15% = 3555 张
    test_size = total - train_size - val_size  # 15% = 3557 张
```

```python
    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices, test_indices = random_split(
        range(total), [train_size, val_size, test_size],
        generator=generator
    )
```
这是 PyTorch 提供的随机划分函数。`manual_seed(42)` 固定随机种子——保证每次运行代码，划分结果都一样。如果不固定种子，每次运行都会得到不同的划分，训练结果就无法复现。

```python
    train_dataset = SubsetDataset(full_dataset, train_indices, get_transforms(is_train=True))
    val_dataset = SubsetDataset(full_dataset, val_indices, get_transforms(is_train=False))
    test_dataset = SubsetDataset(full_dataset, test_indices, get_transforms(is_train=False))
```
用 SubsetDataset 从完整数据集中切出三份，训练集套上数据增强，验证集和测试集只用基础变换。

```python
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
    )
```
- `batch_size=BATCH_SIZE`：每批 128 张
- `shuffle=True`：每个 epoch 随机打乱训练数据顺序，防止模型记住数据顺序
- `num_workers=NUM_WORKERS`：并行进程数
- `pin_memory=True`：锁页内存，加速 CPU→GPU 数据传输
- `drop_last=False`：不丢弃最后一个不完整的 batch（如果总数不能被 batch_size 整除）

验证集和测试集设为 `shuffle=False`，因为评估不需要打乱顺序。

### 3.6 if __name__ == "__main__" 块（第 219-226 行）

```python
if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats = create_dataloaders()
    images, genders, ages = next(iter(train_loader))
    print(f"\n[检查] 一个 batch 的形状: images={images.shape}...")
```

`if __name__ == "__main__":` 是 Python 的惯用写法：这个代码块**只有当直接运行 `python dataset.py` 时才会执行**，而当 `dataset.py` 被其他文件 `import` 时不会执行。

这就像一个自检开关：你可以单独运行 `python dataset.py` 来验证数据加载是否正常，而不需要启动完整训练。

---

## 第四章：model.py 逐行讲解

`model.py` 定义了神经网络的结构——这是整个项目最核心的部分。

### 4.1 导入的库

```python
import torch
import torch.nn as nn
from torchvision import models
```
- `torch.nn`（简写 `nn`）：PyTorch 的神经网络模块，包含各种层类型（Linear、Conv2d、BatchNorm 等）
- `torchvision.models`：PyTorch 官方提供的预训练模型库，包含 ResNet、MobileNet 等

### 4.2 MultiTaskModel 类 —— 构造函数（第 37-69 行）

```python
class MultiTaskModel(nn.Module):
```
`nn.Module` 是 PyTorch 所有神经网络的基类。继承它之后，你只需要定义 `__init__`（搭结构）和 `forward`（定义数据怎么流），PyTorch 会自动处理反向传播、参数管理、模型保存等一切琐事。

```python
def __init__(self, backbone_name=BACKBONE, pretrained=PRETRAINED):
    super().__init__()
```
- `super().__init__()`：调用父类（nn.Module）的构造函数。**这行不能省略**，否则 PyTorch 无法正确注册模型的各个层。
- 参数给了默认值（从 config 读取），你也可以在创建模型时手动覆盖

```python
    self.backbone, self.feature_dim = self._build_backbone(backbone_name, pretrained)
```
调用自己的私有方法 `_build_backbone()`，返回两个东西：
1. `self.backbone`：整个 ResNet 骨干（但最后一层已被替换）
2. `self.feature_dim`：骨干输出的特征向量长度（ResNet18 是 512）

```python
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
```
这就是性别的「分类头」。`nn.Sequential` 是一个容器，把多个层按顺序串起来，数据从左到右依次流过。我们逐层看：

| 层 | 作用 | 输入 → 输出 |
|---|---|---|
| `nn.Linear(512, 256)` | 全连接层，从特征空间映射到 256 维 | 512 → 256 |
| `nn.BatchNorm1d(256)` | 批归一化，稳定训练、加速收敛 | 256 → 256 |
| `nn.ReLU(inplace=True)` | 激活函数，引入非线性 | 256 → 256 |
| `nn.Dropout(p=0.4)` | 随机丢弃 40% 神经元，防止过拟合 | 256 → 256 |
| `nn.Linear(256, 128)` | 再次压缩维度 | 256 → 128 |
| `nn.BatchNorm1d(128)` | 批归一化 | 128 → 128 |
| `nn.ReLU(inplace=True)` | 激活函数 | 128 → 128 |
| `nn.Dropout(p=0.3)` | 随机丢弃 30% | 128 → 128 |
| `nn.Linear(128, 2)` | 最终输出层，输出 2 个分数 | 128 → 2 |

**几个层的解释：**

- **Linear（全连接层）**：`nn.Linear(in_features, out_features)`。它做的事情就是 `y = x·W + b`。每个输出神经元连接到每一个输入神经元。

- **BatchNorm1d（批归一化）**：把每一层的输出标准化到均值 ≈ 0、标准差 ≈ 1。它像一位「格式化助手」，确保数据在每一层之间传递时保持相似的尺度，让训练更稳定、更快。

- **ReLU（Rectified Linear Unit）**：`f(x) = max(0, x)`。把所有负值变成 0，正值保持不变。它是最常用的激活函数。

  **为什么需要激活函数？** 如果没有激活函数（或者说激活函数是 `f(x)=x`），不管堆多少层，整个网络等价于一个线性变换 `y = x·W + b`，无法学习复杂的非线性模式。激活函数打破了这种线性，让深层网络真正有意义。

- **Dropout（随机失活）**：训练时随机把一部分神经元的输出设为 0。前面的层 dropout 率 0.4（丢 40%），后面降到 0.3（丢 30%）。

```python
    self.age_head = nn.Sequential(
        nn.Linear(self.feature_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(128, NUM_AGE_CLASSES),  # ← 117类输出 (0-116岁)
    )
```
年龄头结构和性别头完全对称，只有最后一层不同：
- 性别最后一层输出 2 个值（男和女的得分）
- 年龄最后一层输出 **117 个值**（0 到 116 岁每个年龄的得分）

这 117 个分数经过 Softmax 后变成概率分布，然后做**期望加权求和**得到最终的年龄预测值：
```python
expected_age = sum(softmax(logits)[i] × i  for i in 0..116)
```
比如模型认为有 20% 概率是 30 岁、60% 概率是 31 岁、20% 概率是 32 岁，
期望年龄 = 0.2×30 + 0.6×31 + 0.2×32 = 31.0 岁。

这就是 DEX（Deep EXpectation）方法，既保留了分类的稳定性，又通过加权求和得到连续的年龄值。

```python
    self._init_heads()
```
调用权重初始化方法，给新加的全连接层设置合理的初始值。

### 4.3 _build_backbone() 方法（第 71-97 行）

```python
def _build_backbone(self, name, pretrained):
```
方法名前面的 `_` 是 Python 的命名约定，表示「这是一个内部使用的方法，外部不应该直接调用」。

```python
    if name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        feature_dim = 512
        model.fc = nn.Identity()
```

这四行是核心。以 ResNet18 为例：
1. `models.resnet18(weights=...)`：从 torchvision 加载 ResNet18。如果 `pretrained=True`，就用 ImageNet 预训练权重；如果是 `False`，就随机初始化。
2. `feature_dim = 512`：ResNet18 最后一个卷积层的输出是 512 个通道，经过全局平均池化后得到 512 维的特征向量。
3. `model.fc = nn.Identity()`：这是最关键的一步！ResNet18 原本的 `fc` 层是 `Linear(512, 1000)`，用来对 1000 个 ImageNet 类别做分类。我们把它替换成 `nn.Identity()`。

**`nn.Identity()`** 是什么？它是一个「直通层」——输入什么就输出什么，完全不做任何处理。你可以把它理解为一个透明的玻璃管，数据怎么进就怎么出。

为什么要替换？因为我们不需要 ResNet 原有的 1000 分类能力，我们只需要它提取的特征向量（512 维），然后送入我们自己设计的性别头和年龄头。

对于其他骨干网络，逻辑完全相同：
- ResNet34：和 ResNet18 结构一样，只是层数更多，输出也是 512 维
- ResNet50：更深的网络，使用瓶颈（bottleneck）设计，输出 2048 维
- MobileNetV2：轻量级网络，去掉了最后的 `classifier` 层（不是 `fc`），输出 1280 维

### 4.4 _init_heads() 方法（第 99-105 行）

```python
def _init_heads(self):
    for head in [self.gender_head, self.age_head]:
        for m in head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
```

权重初始化非常重要。如果初始值太大，梯度会爆炸；如果初始值太小，梯度会消失。

- `head.modules()`：递归遍历 head 中的所有子层
- `isinstance(m, nn.Linear)`：只对全连接层做初始化
- `nn.init.kaiming_normal_()`：Kaiming 初始化（也叫 He 初始化），专为 ReLU 激活函数设计的初始化方法，能让信号在深层网络中保持合适的尺度
- `nn.init.constant_(m.bias, 0)`：把偏置全部初始化为 0

**注意：** 我们只初始化两个 head（性别头和年龄头），骨干网络不需要初始化——它已经加载了 ImageNet 预训练权重。

### 4.5 forward() 方法（第 108-121 行）

```python
def forward(self, x):
    features = self.backbone(x)
    gender_out = self.gender_head(features)
    age_out = self.age_head(features)
    return gender_out, age_out
```

`forward()` 定义了数据在模型中的流动路径。**你不需要手动调用它**——当你写 `model(images)` 时，PyTorch 会自动调用 `forward`。

执行流程：
1. `x`（一批图片）进入 ResNet 骨干 → 输出 512 维特征向量
2. 特征向量同时送入性别头和年龄头
3. 性别头输出 `(batch_size, 2)` 的张量（每张图片两个分数）
4. 年龄头输出 `(batch_size, 117)` 的 logits（每张图片 117 个年龄类别的得分）
5. 返回两个结果

之后在 train.py 中，年龄 logits 会经过两个处理：
- 直接与年龄标签计算 **CE 损失**（分类）
- 经 Softmax → 期望加权求和 → 与真实年龄计算 **MAE 损失**（距离）

模型新增了一个工具函数 `compute_expected_age()` 专门做这个加权求和。

### 4.6 build_model() 工厂函数（第 124-137 行）

```python
def build_model():
    from config import DEVICE
    model = MultiTaskModel()
    model = model.to(DEVICE)
```
`.to(DEVICE)` 把模型转移到 GPU（如果可用）或 CPU。所有后续计算都会在指定设备上运行。

```python
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
```
- `model.parameters()`：返回模型中所有参数的迭代器
- `p.numel()`：返回这个参数张量中的元素个数（number of elements）
- `p.requires_grad`：这个参数是否需要在训练中被更新

为什么需要区分？某些高级技巧会把部分参数「冻结」（设置 `requires_grad=False`），只训练其他部分。虽然本项目没有冻结参数，但打印出这个信息有助于你未来做更复杂的实验。

### 4.7 get_loss_functions() 函数

```python
def get_loss_functions():
    gender_criterion = nn.CrossEntropyLoss()
    return gender_criterion
```

- `nn.CrossEntropyLoss()`：交叉熵损失，PyTorch 中做分类任务的标准选择。它内部自动包含了 Softmax 操作，所以模型输出不需要先过 Softmax。
- 注意：年龄的损失函数不在 model.py 中定义，而是在 train.py 中手动组合 CE + MAE，因为需要先从 117 类 logits 算出期望年龄再求 MAE，PyTorch 没有现成的联合损失函数。

此外，model.py 新增了 `compute_expected_age()` 工具函数，从 117 类 logits 计算期望年龄：
```python
def compute_expected_age(age_logits):
    probs = torch.softmax(age_logits, dim=1)
    indices = torch.arange(age_logits.size(1), ...)
    return (probs * indices).sum(dim=1)
```

---

## 第五章：train.py 逐行讲解

`train.py` 是「执行层」——它不定义模型结构，而是定义「怎么训练」「怎么评估」「怎么记录」。

### 5.1 辅助函数

#### compute_age_bin_accuracy()（第 23-41 行）

```python
def compute_age_bin_accuracy(pred_ages, true_ages, bins=AGE_BINS):
    pred_bins = np.digitize(pred_ages, bins) - 1
    true_bins = np.digitize(true_ages, bins) - 1
    pred_bins = np.clip(pred_bins, 0, len(bins) - 1)
    true_bins = np.clip(true_bins, 0, len(bins) - 1)
    return (pred_bins == true_bins).mean()
```

- `np.digitize(x, bins)`：返回 x 中的每个元素落在 bins 的哪个区间。比如 `bins=[0,10,20,30]`，值 23 落在 [20,30) → 返回索引 3
- 减 1 是因为 `digitize` 返回的是 1-based 索引，我们要 0-based
- `np.clip(..., 0, len(bins)-1)`：把超出范围的索引截断到合法范围内
- `(pred_bins == true_bins).mean()`：计算落在同一区间的比例

**例子：** 预测 23 岁，真实 28 岁。bins = [0,10,20,30,40,...]。
- 23 → 区间索引 2（[20,30)）
- 28 → 区间索引 2（[20,30)）
- 两者相等 → 算预测正确！虽然差了 5 岁，但在「年龄段」层面是可以接受的。

#### format_time()（第 44-47 行）

```python
def format_time(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
```
- `divmod(a, b)`：同时返回 `a//b` 和 `a % b`（商和余数）
- `f"{m:02d}"`：格式化成至少 2 位数字，不够前面补 0。比如 `3` → `"03"`

### 5.2 train_one_epoch() —— 训练一轮（第 54-131 行）

这是整个项目最长的函数，但逻辑非常清晰。我们一步一步来。

```python
def train_one_epoch(model, dataloader, gender_criterion, optimizer, epoch):
```
- `model`：神经网络模型
- `dataloader`：训练数据加载器
- `gender_criterion`：性别损失函数（年龄损失在函数内部手动计算 CE+MAE）
- `optimizer`：优化器
- `epoch`：当前是第几轮（仅用于打印）

```python
    model.train()
```
**这一行非常关键！** 它把模型切换到「训练模式」。在这种模式下：
- Dropout 层会正常工作（随机丢弃神经元）
- BatchNorm 层会更新统计信息

如果你忘记调用 `model.train()`，Dropout 在训练时不会生效，模型很容易过拟合。

```python
    total_gender_loss = 0.0
    total_age_ce = 0.0     # CE 分类损失累加
    total_age_mae = 0.0    # MAE 距离损失累加
    total_age_loss = 0.0   # 联合损失累加
    total_loss = 0.0
    correct_gender = 0
    total_samples = 0
    all_pred_ages = []
    all_true_ages = []
```
初始化所有累加器。年龄用联合损失 CE+MAE，所以需要分别累加 CE 和 MAE 两项，方便后续分析每项各自的变化趋势。

```python
    for batch_idx, (images, genders, ages) in enumerate(dataloader):
```
这是训练的核心循环。`enumerate(dataloader)` 每次返回：
- `batch_idx`：当前是第几个 batch（0, 1, 2, ...）
- `(images, genders, ages)`：一个 batch 的数据三元组

```python
        images = images.to(DEVICE)
        genders = genders.to(DEVICE)
        ages = ages.to(DEVICE).float()  # (B,) 浮点年龄，不做 unsqueeze
        age_class_labels = ages.round().long().clamp(0, NUM_AGE_CLASSES - 1)
```
`.to(DEVICE)` 把数据从 CPU 内存移到 GPU 显存。`ages` 保持为 `(B,)` 形状的浮点值，同时生成整数类别标签用于 CE 损失——比如真实年龄 35.0 就对应第 35 类。`clamp(0, 116)` 确保超出 0-116 范围的年龄被截断（UTKFace 数据集中有极少数 >116 岁的样本）。

```python
        gender_logits, age_logits = model(images)
```
**前向传播**。把图片送入模型，拿到性别 logits (B,2) 和年龄 logits (B,117)。

**logits 是什么？** logits 就是模型最后一层输出的原始分数，还没有经过 Softmax。比如 `[-3.2, 2.1]` 表示模型认为类别 1（女性）的分数更高。CrossEntropyLoss 内部会自动处理 Softmax，所以不需要我们手动做。

```python
        loss_g = gender_criterion(gender_logits, genders)

        # 年龄联合损失: CE + λ × MAE
        loss_age_ce = F.cross_entropy(age_logits, age_class_labels)
        expected_age = compute_expected_age(age_logits)    # Softmax加权求和
        loss_age_mae = F.l1_loss(expected_age, ages)       # L1距离
        loss_a = loss_age_ce + AGE_CE_LAMBDA * loss_age_mae

        loss = loss_g + loss_a
```
年龄损失的计算分三步：
1. **CE 损失**：117 类交叉熵，把年龄当分类问题。比如「这人是 35 岁」→ 模型必须给第 35 类打最高分
2. **期望年龄**：`compute_expected_age()` 对 softmax 概率做加权求和，得到连续的年龄值。比如模型认为 30% 概率 34 岁 + 50% 概率 35 岁 + 20% 概率 36 岁 → 期望 = 0.3×34 + 0.5×35 + 0.2×36 = 34.9 岁
3. **MAE 损失**：期望年龄与真实年龄的 L1 距离，用 `AGE_CE_LAMBDA = 0.5` 加权后与 CE 求和

```python
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
这是训练的核心三行，每次 batch 执行一遍：

1. **`optimizer.zero_grad()`**：清空上一次迭代计算的梯度。PyTorch 默认会**累加**梯度（为了支持某些高级技巧），所以每次反向传播前必须手动清零。如果不做这步，梯度会越累积越大，训练完全跑偏。

2. **`loss.backward()`**：反向传播。PyTorch 自动计算损失相对于每个参数的梯度（偏导数）。

3. **`optimizer.step()`**：优化器根据梯度更新参数。比如参数 = 参数 - 学习率 × 梯度。

这三行的顺序绝对不能错：先清零，再算梯度，最后更新。

```python
        bs = images.size(0)
        total_samples += bs
        total_gender_loss += loss_g.item() * bs
        total_age_ce += loss_age_ce.item() * bs
        total_age_mae += loss_age_mae.item() * bs
        total_age_loss += loss_a.item() * bs
        total_loss += loss.item() * bs
```
- `bs`：当前 batch 的实际大小（通常是 128，最后一个 batch 可能更少）
- `loss_g.item()`：把 PyTorch 的张量转成 Python 的浮点数
- 乘以 `bs` 是为了加权平均：batch 越大，对总损失的贡献应该越大
- 年龄的 CE 和 MAE 分开累加，方便在 epoch 结束后分析各项的贡献

```python
        _, predicted_gender = torch.max(gender_logits, 1)
        correct_gender += (predicted_gender == genders).sum().item()
```
- `torch.max(gender_logits, 1)`：在维度 1（类别维度）上取最大值。返回两个东西：最大值本身（我们不要，用 `_` 接收）和最大值的索引（即预测的类别）。比如 logits 是 `[-3.2, 2.1]`，最大值索引是 `1`（女性）。
- `(predicted_gender == genders).sum().item()`：统计这个 batch 中预测正确的数量

```python
        all_pred_ages.extend(expected_age.detach().cpu().numpy().flatten())
        all_true_ages.extend(ages.detach().cpu().numpy().flatten())
```
- `.detach()`：把张量从计算图中分离出来（我们只是收集统计信息，不需要追踪梯度）
- `.cpu()`：把数据从 GPU 移回 CPU
- `.numpy()`：转成 NumPy 数组
- `.flatten()`：展平成一维数组

```python
        if batch_idx % 100 == 0 and batch_idx > 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    Batch [{batch_idx:4d}/{len(dataloader)}] ...")
```
每 100 个 batch 打印一次进度。`% 100 == 0` 表示能被 100 整除。条件 `batch_idx > 0` 确保 batch 0 时不打印（此时模型还是初始状态）。

**epoch 结束后的汇总计算：**

```python
    avg_gender_loss = total_gender_loss / total_samples
    avg_age_ce = total_age_ce / total_samples
    avg_age_mae = total_age_mae / total_samples
    avg_age_loss = total_age_loss / total_samples
    avg_total_loss = total_loss / total_samples
    gender_acc = correct_gender / total_samples * 100
```
用累积的总损失除以总样本数，得到 epoch 级别的平均损失。准确率乘以 100 转成百分比。

```python
    all_pred_ages = np.array(all_pred_ages)
    all_true_ages = np.array(all_true_ages)
    age_mae = np.abs(all_pred_ages - all_true_ages).mean()
    age_rmse = np.sqrt(((all_pred_ages - all_true_ages) ** 2).mean())
    age_bin_acc = compute_age_bin_accuracy(all_pred_ages, all_true_ages) * 100
```
三种年龄评估指标：
- **MAE（Mean Absolute Error）**：`|预测-真实|` 的平均值。最直观——「平均差几岁」
- **RMSE（Root Mean Squared Error）**：`(预测-真实)²` 的均值再开方。对大误差更敏感
- **年龄分箱准确率**：前面讲过了

### 5.3 evaluate() —— 评估一轮（第 138-197 行）

```python
@torch.no_grad()
def evaluate(model, dataloader, gender_criterion):
```

`@torch.no_grad()` 是装饰器（decorator），它告诉 PyTorch：「下面这个函数里的所有操作都不需要追踪梯度」。这样做两个好处：
1. 节省 GPU 显存（不需要存储中间结果用于反向传播）
2. 加速计算

```python
    model.eval()
```
**把模型切换到评估模式！** 和 `model.train()` 相反：
- Dropout 层停止工作（不再丢弃神经元）
- BatchNorm 层使用训练时积累的全局统计信息，而不是当前 batch 的统计信息

如果你在评估时忘记调用 `model.eval()`，得到的结果会不准确（Dropout 随机丢弃神经元会让每次评估结果都不一样）。

`evaluate()` 的其余逻辑和 `train_one_epoch()` 几乎一样，区别只有：
- 没有 `optimizer.zero_grad() / loss.backward() / optimizer.step()`（评估时不更新参数）
- 没有 `model.train()`，而是 `model.eval()`
- 没有 `age_criterion` 参数（年龄的 CE+MAE 在函数内部手动计算）
- 没有 batch 进度打印

### 5.4 log_epoch() —— 记录日志（第 204-262 行）

```python
def log_epoch(epoch, train_metrics, val_metrics, lr, elapsed):
```

这个函数做两件事：**写 CSV 文件** + **打印到控制台**。

**CSV 写入部分：**

```python
    row = {
        "epoch": epoch,
        "lr": lr,
        "train_gender_loss": f"{train_metrics['gender_loss']:.6f}",
        ...
    }
```
`:.6f` 表示保留 6 位小数。每个 epoch 都追加一行到 CSV 文件中，方便训练结束后用 Excel 或 Python 画图分析。

```python
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)
```
- `"a"` 模式：以追加模式打开文件（append），不会覆盖之前的内容
- `csv.DictWriter`：用字典格式写入 CSV。第一次写之前先写表头（`writeheader()`）

**控制台输出部分：**

```python
    label_map = {
        "Gender Loss":  ("train_gender_loss", "val_gender_loss", ".6f"),
        "Age CE":       ("train_age_ce", "val_age_ce", ".6f"),
        "Age MAE Loss": ("train_age_mae_loss", "val_age_mae_loss", ".6f"),
        "Age Loss":     ("train_age_loss", "val_age_loss", ".6f"),
        "Total Loss":   ("train_total_loss", "val_total_loss", ".6f"),
        "Gender Acc":   ("train_gender_acc", "val_gender_acc", ".2f", "%"),
        "Age MAE":      ("train_age_mae", "val_age_mae", ".2f", " yrs"),
        "Age RMSE":     ("train_age_rmse", "val_age_rmse", ".2f", " yrs"),
        "Age Bin Acc":  ("train_age_bin_acc", "val_age_bin_acc", ".2f", "%"),
    }
```
现在年龄损失分成了三行输出：
- **Age CE**：117 类交叉熵损失，反映「分类能力」
- **Age MAE Loss**：期望年龄与真实年龄的 L1 损失，反映「距离感知」
- **Age Loss**：两者的加权和

CSV 中也相应增加了 `train_age_ce`、`val_age_ce` 等列，方便后续分析两项各自的变化趋势。

```python
    lines = [
        f"+{'-'*78}+",
        f"| Epoch {epoch:3d}  |  LR: {lr:.2e}  |  Time: {elapsed}",
        ...
    ]
```
Python 的 f-string 格式化语法：
- `{'-'*78}`：78 个减号，形成一条线
- `{epoch:3d}`：3 位整数，不够补空格
- `{lr:.2e}`：科学计数法保留 2 位小数，如 `1.00e-04`

最终打印的效果像一张干净整洁的表格（你之前训练时看到的那个）。

---

## 第六章：main.py 逐行讲解

`main.py` 是整个项目的入口。相比旧版本，现在增加了**分阶段训练**和**差异化学习率**两个关键策略。

### 6.1 文件头与导入

```python
import time
import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from config import (
    DEVICE, EPOCHS, WEIGHT_DECAY,
    BACKBONE_LR, GENDER_HEAD_LR, AGE_HEAD_LR,  # 三个不同学习率
    STAGE_SPLIT,                                 # 阶段划分比例
    ...
)
from model import build_model, get_loss_functions
from train import train_one_epoch, evaluate, log_epoch, format_time
```

### 6.2 辅助函数：freeze_module 和 build_optimizer

```python
def freeze_module(module, freeze=True):
    for param in module.parameters():
        param.requires_grad = not freeze
```

`freeze_module()` 控制一个模块是否参与训练。设置为 `freeze=True` 后，该模块所有参数的 `requires_grad` 变为 `False`，优化器会跳过它们。

```python
def build_optimizer(model, stage):
    if stage == 1:
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
```

不同阶段，优化器管理的参数组不同：
- **阶段一**：只有骨干 + 年龄头（性别头冻结，不需要优化器管理）
- **阶段二**：骨干 + 性别头 + 年龄头，三个组件各用不同的学习率

| 组件 | 学习率 | 原因 |
|------|--------|------|
| 骨干 (ResNet18) | `1e-5` | ImageNet 预训练权重已经很好，只需微调 |
| 性别头 | `1e-4` | 从头训练，但二分类简单，中等 LR 即可 |
| 年龄头 | `2e-4` | 从头训练 117 类分类，任务最复杂，需要最大 LR |

### 6.3 main() 函数 —— 分阶段训练

整个训练分为**两个阶段**：

**阶段一（Epoch 1 - 15）：冻结性别头，只训练年龄头**
```python
    stage1_epochs = int(EPOCHS * STAGE_SPLIT)   # 15 epoch

    freeze_module(model.gender_head, freeze=True)   # 冻结性别头
    optimizer = build_optimizer(model, stage=1)     # 只优化骨干+年龄头

    for epoch in range(1, stage1_epochs + 1):
        train_metrics = train_one_epoch(...)
        val_metrics = evaluate(...)
        ...
```

这个阶段的目的：**让年龄头先学好**。性别是简单任务（二分类，很快就能 90%+），如果从一开始就和年龄一起训练，性别会抢走共享骨干的大部分梯度信号，年龄头就很难学到细粒度特征。先冻结性别头，让骨干网络全力配合年龄头学习皱纹、皮肤纹理等年龄相关特征。

**阶段二（Epoch 16 - 30）：解冻性别头，全参数联合训练**
```python
    freeze_module(model.gender_head, freeze=False)   # 解冻
    optimizer = build_optimizer(model, stage=2)      # 三个组件各用不同 LR

    for epoch in range(stage1_epochs + 1, EPOCHS + 1):
        train_metrics = train_one_epoch(...)
        ...
```

这个阶段的目的：**联合微调**。年龄头已经有了不错的基础，现在让性别头加入，三个组件用差异化学习率一起训练。骨干网络用最低的学习率（保护预训练特征），年龄头维持较高学习率，性别头用中等学习率从头追赶。

**为什么不用一个 for 循环分两个 if 判断？**

两个阶段之间的切换需要重建 optimizer（参数组从 2 组变成 3 组），所以用两个独立的 for 循环更清晰。阶段切换时打印分隔线，日志中也能清楚看到哪个 epoch 属于哪个阶段。
- `weights_only=False`：允许加载完整的 checkpoint（包含优化器状态等），而不仅仅是模型权重
- `model.load_state_dict(...)`：把保存的权重恢复到模型中

```python
    test_metrics = evaluate(model, test_loader, gender_criterion)
```
在测试集上做最终评估。注意这里用的是测试集的 DataLoader，而之前训练和验证阶段从没碰过。

```python
    print(f"\n  ┌{'─'*36}┐")
    print(f"  │ {'测试集最终结果':^34s} │")
    ...
```
用 Unicode 框线字符（`┌ ├ └ ┤ ─ │`）画一个漂亮的表格来展示最终结果。

### 6.3 if __name__ == "__main__"（第 155-156 行）

```python
if __name__ == "__main__":
    main()
```
当你运行 `python main.py` 时，Python 解释器会把 `__name__` 设为 `"__main__"`，于是执行 `main()`。如果 `main.py` 被其他文件 import，这个条件为 False，`main()` 不会被自动执行。

---

## 第七章：运行与理解输出

### 7.1 启动训练

```bash
python main.py
```

### 7.2 输出解读

训练开始时会打印：
```
============================================================
   MultiFace — 人脸性别与年龄预测项目
============================================================
  数据目录:     D:\MultiFace\data\UTKFace
  图像尺寸:     224×224
  批次大小:     128
  骨干网络:     resnet18 (pretrained=True)
  训练设备:     cuda
  ...
```
这是 `print_config()` 打印的配置摘要，确认一切设置符合预期。

然后打印数据统计：
```
[数据集] 发现 23708 个图片文件
[数据集] 有效样本: 23705
[数据集] 男性: 12391, 女性: 11314
[数据集] 年龄: 均值=33.3, 标准差=19.9, 范围=[1, 116]
[数据集] 训练集: 16593, 验证集: 3555, 测试集: 3557
```

然后每个 epoch 打印一张表格：
```
+------------------------------------------------------------------------------+
| Epoch   1  |  LR: 1.00e-04  |  Time: 01:32
+---------------------------------------+---------------------------------------+
| Metric                           Train |               Val |
+---------------------------------------+---------------------------------------+
| Gender Loss          0.561654      | 0.276530      |
| Age Loss             10.238573      | 8.225826      |
| Total Loss           4.657083      | 3.566861      |
| Gender Acc           86.39    % | 90.83    % |
| Age MAE              10.24  yrs | 8.23  yrs |
| Age RMSE             15.36  yrs | 20.55  yrs |
| Age Bin Acc          39.56    % | 48.69    % |
+------------------------------------------------------------------------------+
```

**怎么读这张表：**
- **Loss 系列**：越小越好，说明预测和真实值差距小
- **Gender Acc**：越高越好，说明男女分得准
- **Age MAE/RMSE**：越小越好，说明年龄预测得准
- **Age Bin Acc**：越高越好，说明年龄段判断准确
- **Train vs Val**：如果 Train 远好于 Val → 可能过拟合；如果两者接近 → 模型泛化能力好

### 7.3 训练完成后

最终测试集结果：
```
  ┌────────────────────────────────────┐
  │          测试集最终结果              │
  ├────────────────────────────────────┤
  │ 性别准确率 (Gender Acc):    91.23%  │
  │ 年龄 MAE (Age MAE):          7.15 yrs │
  │ 年龄 RMSE (Age RMSE):        18.42 yrs │
  │ 年龄分箱准确率 (Bin Acc):    55.67%  │
  └────────────────────────────────────┘
```

产出文件：
- `output/best_model.pth`：最佳模型权重，可用于后续推理
- `output/training_log.csv`：每轮详细指标，可用 Excel 打开画曲线图

---

## 附录：代码阅读顺序建议

如果你是零基础，建议按以下顺序阅读源代码：

1. **`config.py`**（最简单，纯参数定义）
2. **`model.py` 的 `get_loss_functions()`**（理解损失函数）
3. **`dataset.py` 的 `UTKFaceDataset`**（理解数据长什么样）
4. **`model.py` 的 `MultiTaskModel`**（理解模型结构）
5. **`train.py` 的 `train_one_epoch`**（理解一轮训练做什么）
6. **`train.py` 的 `evaluate`**（理解评估怎么做）
7. **`main.py` 的 `main()`**（理解整个流程）
8. 回头再看一遍全部代码，这次应该能完全理解了

记住：**读代码就像读一本小说，第一遍不需要理解每一个字**。先把握整体脉络，再深入细节。如果某个概念卡住了，回到本文第〇章查对应的概念讲解。

---

*本文档面向深度学习零基础读者编写。如有任何概念或代码段落需要进一步解释，欢迎继续提问。*
