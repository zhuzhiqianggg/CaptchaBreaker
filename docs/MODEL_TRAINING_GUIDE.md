# 大模型训练技术文档

## 一、大模型训练概述

### 1.1 什么是大模型训练

大模型训练（Large Model Training）是指使用大量数据和计算资源，训练具有数十亿到数千亿参数的深度学习模型。常见的应用场景包括：

- **自然语言处理（NLP）**：GPT、BERT、LLaMA 等语言模型
- **计算机视觉（CV）**：ViT、CLIP、SAM 等视觉模型
- **多模态模型**：GPT-4V、Gemini、Qwen-VL 等
- **OCR 专用模型**：PaddleOCR、TrOCR 等

### 1.2 训练流程总览

```
数据收集 → 数据清洗 → 模型选择 → 训练 → 评估 → 部署 → 持续优化
```

---

## 二、训练核心概念

### 2.1 训练类型

| 训练类型 | 说明 | 适用场景 | 资源需求 |
|---------|------|---------|---------|
| **预训练（Pre-training）** | 从随机初始化开始训练 | 构建基础模型 | 极高（千卡 GPU 集群） |
| **微调（Fine-tuning）** | 在预训练模型基础上继续训练 | 适应特定领域 | 中等（单卡/多卡 GPU） |
| **迁移学习（Transfer Learning）** | 使用已有模型迁移到新任务 | 数据量少的场景 | 低（单卡 GPU） |
| **增量训练（Incremental Training）** | 在已有模型上追加新数据 | 数据持续增加 | 低-中 |

### 2.2 关键指标

- **Loss（损失函数）**：衡量模型预测与真实值的差距，越低越好
- **Accuracy（准确率）**：正确预测的比例
- **Precision/Recall**：精确率和召回率
- **F1 Score**：精确率和召回率的调和平均
- **BLEU/ROUGE**：文本生成任务的评价指标

### 2.3 超参数

| 超参数 | 说明 | 常用值 |
|--------|------|--------|
| **Learning Rate（学习率）** | 参数更新的步长 | 1e-3 ~ 1e-5 |
| **Batch Size（批次大小）** | 每次训练的样本数 | 16 ~ 256 |
| **Epoch（训练轮数）** | 完整遍历数据集的次数 | 5 ~ 50 |
| **Optimizer（优化器）** | 参数更新策略 | Adam, SGD, AdamW |
| **Scheduler（学习率调度）** | 动态调整学习率 | Cosine, Step, Warmup |

---

## 三、训练环境搭建

### 3.1 硬件要求

#### CPU 训练
- 适用场景：小规模模型、推理测试
- 推荐配置：16 核+ CPU，64GB+ 内存

#### GPU 训练（推荐）
- 适用场景：所有大模型训练
- 推荐配置：
  - **入门级**：RTX 3060/4060（8-12GB 显存）
  - **中端级**：RTX 4080/4090（16-24GB 显存）
  - **企业级**：A100/H100（40-80GB 显存）

### 3.2 软件环境

```bash
# Python 环境（推荐 conda）
conda create -n training python=3.10
conda activate training

# 深度学习框架
pip install torch torchvision  # PyTorch
# 或
pip install paddlepaddle-gpu   # PaddlePaddle

# 训练工具
pip install transformers datasets
pip install accelerate
pip install tensorboard

# 可视化工具
pip install tensorboard
pip install wandb  # Weights & Biases
```

### 3.3 Docker 环境（推荐用于生产）

```yaml
services:
  training:
    image: nvidia/cuda:11.8.0-devel-ubuntu22.04
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./data:/data
      - ./models:/models
    command: bash -c "pip install -r requirements.txt && python train.py"
```

---

## 四、数据准备

### 4.1 数据收集

```python
# 数据收集示例
import os
from pathlib import Path

def collect_data(source_dir, target_dir, min_samples=1000):
    """收集并整理训练数据"""
    collected = 0
    
    for file in Path(source_dir).glob("*"):
        if file.suffix in ['.png', '.jpg', '.jpeg']:
            # 复制到训练数据目录
            target = Path(target_dir) / f"sample_{collected:06d}{file.suffix}"
            file.copy(target)
            collected += 1
    
    print(f"Collected {collected} samples")
    return collected >= min_samples
```

### 4.2 数据清洗

```python
from PIL import Image
import os

def clean_dataset(data_dir, min_size=(20, 20), max_size=(500, 200)):
    """清洗数据集，移除无效样本"""
    removed = 0
    
    for file in Path(data_dir).glob("*"):
        try:
            img = Image.open(file)
            w, h = img.size
            
            # 检查尺寸
            if w < min_size[0] or w > max_size[0]:
                os.remove(file)
                removed += 1
            elif h < min_size[1] or h > max_size[1]:
                os.remove(file)
                removed += 1
        except:
            os.remove(file)
            removed += 1
    
    print(f"Removed {removed} invalid samples")
```

### 4.3 数据标注

**OCR 数据标注格式示例（PaddleOCR）**：

```
# rec_gt_train.txt 格式
image_name1.png	验证码文本1
image_name2.png	验证码文本2
```

---

## 五、模型训练流程

### 5.1 完整训练流程

```python
import paddle
from paddleocr import PaddleOCR

# Step 1: 加载预训练模型
model = PaddleOCR(use_angle_cls=True, lang='en')

# Step 2: 配置训练参数
train_config = {
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'save_dir': './output/model',
    'save_interval': 10,  # 每 10 个 epoch 保存一次
    'eval_interval': 5,   # 每 5 个 epoch 评估一次
}

# Step 3: 开始训练
model.train(
    train_data='./data/train',
    eval_data='./data/eval',
    **train_config
)

# Step 4: 导出模型
model.export_model(
    save_path='./output/export',
    inference=True
)
```

### 5.2 训练监控

```python
# 使用 TensorBoard 监控
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./logs')

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    eval_acc = evaluate(model, eval_loader)
    
    # 记录指标
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/eval', eval_acc, epoch)
    
    print(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={eval_acc:.4f}")

writer.close()
```

### 5.3 训练优化技巧

#### 1. 学习率预热（Warmup）
```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,    # 预热步数
    num_training_steps=10000 # 总训练步数
)
```

#### 2. 梯度累积（模拟大 Batch Size）
```python
accumulation_steps = 4
for i, batch in enumerate(train_loader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### 3. 混合精度训练（加速训练）
```python
# PyTorch AMP
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 六、模型评估

### 6.1 评估指标

```python
def evaluate_model(model, test_data):
    """评估模型性能"""
    results = {
        'accuracy': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1_score': 0.0,
        'char_accuracy': 0.0,  # 字符级准确率（OCR 特有）
    }
    
    total = 0
    correct = 0
    char_total = 0
    char_correct = 0
    
    for sample in test_data:
        pred = model.predict(sample['image'])
        expected = sample['label']
        
        # 整体准确率
        if pred == expected:
            correct += 1
        
        # 字符级准确率
        for p_char, e_char in zip(pred, expected):
            char_total += 1
            if p_char == e_char:
                char_correct += 1
        
        total += 1
    
    results['accuracy'] = correct / total
    results['char_accuracy'] = char_correct / char_total
    
    return results
```

### 6.2 评估脚本示例

```bash
# 评估模型
python eval.py \
    --model_path ./output/model/best.pth \
    --test_data ./data/test \
    --batch_size 32
```

---

## 七、模型部署

### 7.1 导出推理模型

```python
# 导出为推理格式（更小更快）
model.save_inference_model(
    save_dir='./output/inference',
    model_name='ocr_model'
)
```

### 7.2 Docker 部署

```yaml
services:
  inference:
    image: ocr-inference:latest
    ports:
      - "8000:8000"
    volumes:
      - ./output/inference:/models
    environment:
      - MODEL_PATH=/models/ocr_model
      - DEVICE=cuda
```

---

## 八、持续训练策略

### 8.1 数据驱动训练

```
用户请求 → 收集数据 → 达到阈值 → 自动训练 → 评估效果 → 部署新模型
```

### 8.2 自动训练脚本

```python
#!/usr/bin/env python3
"""自动训练调度器"""

import schedule
import time
from pathlib import Path

def check_and_train():
    """检查数据量并触发训练"""
    data_count = len(list(Path('./data/collected').glob('*.png')))
    
    if data_count >= 1000:  # 达到训练阈值
        print(f"Collected {data_count} samples, starting training...")
        
        # 1. 准备数据
        os.system('python scripts/prepare_training.py')
        
        # 2. 训练模型
        os.system('python scripts/train_model.py')
        
        # 3. 评估模型
        os.system('python scripts/evaluate_model.py')
        
        # 4. 部署模型（如果效果更好）
        os.system('python scripts/deploy_model.py')
    else:
        print(f"Only {data_count} samples collected, waiting...")

# 每天检查一次
schedule.every().day.at("02:00").do(check_and_train)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 每小时检查一次
```

---

## 九、常见问题排查

### 9.1 训练不收敛

**症状**：Loss 不下降或波动剧烈

**解决方案**：
1. 降低学习率（例如 1e-4 → 1e-5）
2. 增加 Batch Size
3. 检查数据标注是否正确
4. 使用学习率预热

### 9.2 显存不足（OOM）

**解决方案**：
1. 减小 Batch Size
2. 启用混合精度训练
3. 使用梯度累积
4. 使用梯度检查点（Gradient Checkpointing）

### 9.3 过拟合

**症状**：训练集准确率很高，但测试集很低

**解决方案**：
1. 增加数据量
2. 使用数据增强
3. 添加正则化（Dropout、Weight Decay）
4. 提前停止（Early Stopping）

---

## 十、最佳实践

1. **从小规模开始**：先用 100 个样本测试训练流程
2. **监控训练过程**：使用 TensorBoard 或 WandB
3. **定期保存模型**：每 N 个 epoch 保存一次
4. **保留多个版本**：方便回滚
5. **文档记录**：记录每次训练的超参数和结果
6. **增量训练**：利用已有模型，避免从头训练
7. **A/B 测试**：新旧模型并行测试，确认效果提升后再切换

---

## 参考资料

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [PyTorch 训练教程](https://pytorch.org/tutorials/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [深度学习训练技巧](https://github.com/wkentaro/awesome-deep-learning-papers)
