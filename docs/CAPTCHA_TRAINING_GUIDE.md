# CaptchaBreaker 验证码识别训练技术文档

## 一、项目训练架构概述

### 1.1 技术栈

```
用户请求 → FastAPI 服务 → PaddleOCR 识别 → 数据收集 → 自动训练 → 模型更新
```

**核心技术：**
- **OCR 引擎**：PaddleOCR v5（PP-OCRv5）
- **API 框架**：FastAPI + Uvicorn
- **训练框架**：PaddlePaddle / PaddleX
- **数据处理**：Pillow + NumPy

### 1.2 训练流程总览

```
┌─────────────────────────────────────────────────────────────┐
│                    CaptchaBreaker 训练流程                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户请求 ──→ OCR 识别 ──→ 保存数据 ──→ 达到阈值 ──→ 训练     │
│     │                                                        │
│     ↓                                                        │
│  ┌─────────────┐                                             │
│  │ 数据收集模块 │←── 原始图片 + 4 种预处理 + 识别结果 + 标注    │
│  └──────┬──────┘                                             │
│         ↓                                                    │
│  ┌─────────────┐                                             │
│  │ 训练准备模块 │←── 格式转换 + 字符字典 + 训练集划分           │
│  └──────┬──────┘                                             │
│         ↓                                                    │
│  ┌─────────────┐                                             │
│  │ 模型训练模块 │←── Fine-tuning + 早停 + 保存最佳模型         │
│  └──────┬──────┘                                             │
│         ↓                                                    │
│  ┌─────────────┐                                             │
│  │ 模型评估模块 │←── 准确率测试 + 对比基线                    │
│  └──────┬──────┘                                             │
│         ↓                                                    │
│  ┌─────────────┐                                             │
│  │ 模型部署模块 │←── 热更新 / 灰度发布 + 回滚机制              │
│  └─────────────┘                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、数据收集机制

### 2.1 自动数据收集

系统每次识别都会自动保存训练数据，保存在 `data/collected/` 目录：

```
data/collected/
├── original/          # 原始验证码图片
├── v1_light/          # 轻量预处理后的图片
├── v2_medium/         # 中等增强后的图片
├── v3_binary/         # 二值化处理后的图片
├── v4_denoise/        # 降噪处理后的图片
└── metadata/          # 元数据（JSON 格式）
```

### 2.2 元数据结构

每个识别结果都会生成一个 JSON 元数据文件：

```json
{
  "image_id": "abc123def456",
  "final_text": "2zrw",
  "corrected_text": "2zRw",
  "strategies": {
    "v1_light": {
      "ocr_result": "2zrw",
      "preprocessing_steps": ["v1:Light", "Contrast+1.5x"]
    },
    "v2_medium": {
      "ocr_result": "2zRw",
      "preprocessing_steps": ["v2:Medium", "Contrast+2.0x", "Sharpness+1.5x"]
    },
    "v3_binary": {
      "ocr_result": "2zRW",
      "preprocessing_steps": ["v3:Binarize", "Binary threshold=128"]
    },
    "v4_denoise": {
      "ocr_result": "2zrw",
      "preprocessing_steps": ["v4:Denoise", "Median filter", "Contrast+1.3x"]
    }
  }
}
```

### 2.3 数据收集代码

数据收集在 [save_training_data](file:///c%3A/dev_project/ocr/app/main.py#L106-L159) 函数中实现：

```python
def save_training_data(
    image_id: str,
    original_image: Image.Image,
    processed_images: Dict[str, Tuple[Image.Image, List[str]]],
    ocr_results: Dict[str, str],
    final_text: str,
    corrected_text: str = None
):
    """保存训练数据用于后期模型 fine-tuning"""
    train_dirs = get_train_dirs()
    
    # 保存原始图片
    original_path = os.path.join(train_dirs["original"], f"{image_id}.png")
    original_image.save(original_path)
    
    # 保存预处理后的图片和元数据
    metadata = {
        "image_id": image_id,
        "final_text": final_text,
        "corrected_text": corrected_text,
        "strategies": {}
    }
    
    for strategy_name, (processed_img, steps) in processed_images.items():
        # 保存预处理图片
        save_path = os.path.join(train_dirs[dir_key], f"{image_id}.png")
        processed_img.save(save_path)
        
        # 记录策略结果
        metadata["strategies"][dir_key] = {
            "ocr_result": ocr_results.get(strategy_name, ""),
            "preprocessing_steps": steps
        }
    
    # 保存元数据
    metadata_path = os.path.join(train_dirs["metadata"], f"{image_id}.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
```

---

## 三、训练数据准备

### 3.1 数据格式转换

PaddleOCR 训练需要特定的标注格式。使用 [scripts/prepare_training.py](file:///c%3A/dev_project/ocr/scripts/prepare_training.py) 进行转换：

```bash
# 准备训练数据
python scripts/prepare_training.py
```

**生成文件：**
- `data/finetune/rec_gt_train.txt` - 训练标注文件
- `data/finetune/char_dict.txt` - 字符字典
- `data/finetune/train/` - 训练图片目录

### 3.2 标注文件格式

```
# rec_gt_train.txt 示例
img001.png	2zrw
img002.png	6QxJ
img003.png	78sq
img004.png	aDkP
```

### 3.3 字符字典

```
# char_dict.txt 示例
0
1
2
3
...
A
B
C
...
a
b
c
...
```

### 3.4 数据增强（可选）

为了提高模型泛化能力，可以对训练数据进行增强：

```python
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import random

def augment_image(image, label):
    """数据增强"""
    augmented = []
    
    # 原始图片
    augmented.append((image, label))
    
    # 随机旋转（-5° 到 5°）
    angle = random.uniform(-5, 5)
    rotated = image.rotate(angle, resample=Image.BICUBIC)
    augmented.append((rotated, label))
    
    # 随机对比度调整
    factor = random.uniform(0.8, 1.5)
    enhancer = ImageEnhance.Contrast(image)
    contrasted = enhancer.enhance(factor)
    augmented.append((contrasted, label))
    
    # 随机添加噪声
    noise = np.random.randint(0, 50, image.size, dtype=np.uint8)
    noisy = Image.fromarray(np.array(image) + noise)
    augmented.append((noisy, label))
    
    return augmented
```

---

## 四、模型训练

### 4.1 使用 PaddleX 训练（推荐）

PaddleX 是百度推出的全流程开发工具，支持一键训练：

```bash
# 安装 PaddleX
pip install paddlex

# 训练命令
paddlex --task OCR --train \
    --dataset ./data/finetune \
    --save_dir ./output/finetune \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### 4.2 使用 PaddleOCR 原生训练

```bash
# 1. 克隆 PaddleOCR 仓库
git clone https://github.com/PaddlePaddle/PaddleOCR.git
cd PaddleOCR

# 2. 准备配置文件
cp configs/rec/rec_mv3_en_PP-OCRv5_train.yml configs/rec/rec_custom.yml

# 编辑配置文件，修改以下参数：
# Global:
#   pretrained_model: ./pretrain_models/en_PP-OCRv5_mobile_rec/best_accuracy
#   save_model_dir: ./output/finetune
#   character_dict_path: ../data/finetune/char_dict.txt
# TrainData:
#   data_dir: ../data/finetune/train/
#   label_file_list: [../data/finetune/rec_gt_train.txt]

# 3. 开始训练
python tools/train.py -c configs/rec/rec_custom.yml
```

### 4.3 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **epochs** | 训练轮数 | 100-200 |
| **batch_size** | 批次大小 | 16-64（根据显存调整） |
| **learning_rate** | 学习率 | 1e-4 ~ 5e-4 |
| **optimizer** | 优化器 | Adam |
| **pretrained_model** | 预训练模型路径 | en_PP-OCRv5_mobile_rec |
| **save_interval** | 保存间隔 | 10 |
| **eval_interval** | 评估间隔 | 5 |

### 4.4 训练监控

训练过程中会输出以下信息：

```
Epoch [10/100], Step [100/500], Loss: 0.1234, Acc: 0.85, ETA: 2:30:00
Epoch [20/100], Step [200/500], Loss: 0.0891, Acc: 0.89, ETA: 2:00:00
...
```

**关键指标：**
- **Loss**：损失值，越低越好
- **Acc**：准确率，越高越好
- **ETA**：预计剩余时间

### 4.5 训练最佳实践

1. **使用预训练模型**：从官方预训练模型开始微调，比从头训练快 10 倍
2. **早停机制（Early Stopping）**：验证集准确率连续 10 个 epoch 不提升时停止
3. **保存最佳模型**：保存验证集上准确率最高的模型
4. **多尺度训练**：训练时使用不同尺寸的图片，提高泛化能力

---

## 五、模型评估

### 5.1 评估脚本

```bash
# 评估训练好的模型
python scripts/evaluate_model.py \
    --model_path ./output/finetune/best_accuracy.pdparams \
    --test_data ./data/samples \
    --batch_size 32
```

### 5.2 评估指标

```
====================================
模型评估结果
====================================
整体准确率:    92.5%
字符级准确率:  96.3%
大小写准确率:  88.7%
数字准确率:    98.1%
字母准确率:    89.4%
====================================
```

### 5.3 对比基线

训练完成后，需要对比新模型和旧模型的效果：

| 模型 | 准确率 | 提升 |
|------|--------|------|
| 官方预训练模型 | 85.0% | - |
| Fine-tuned v1 | 92.5% | +7.5% |
| Fine-tuned v2 | 94.2% | +1.7% |

---

## 六、模型部署

### 6.1 导出推理模型

```bash
# 导出为推理格式（更小更快）
python tools/export_model.py \
    -c configs/rec/rec_custom.yml \
    -o Global.pretrained_model=./output/finetune/best_accuracy \
    Global.save_inference_dir=./output/inference
```

### 6.2 部署到服务

```bash
# 1. 备份旧模型
mv models/custom models/custom_backup

# 2. 部署新模型
cp -r ./output/inference models/custom

# 3. 重启服务
docker-compose restart
```

### 6.3 热更新（不重启服务）

修改 [app/main.py](file:///c%3A/dev_project/ocr/app/main.py) 添加模型热更新功能：

```python
@app.post("/admin/reload_model")
async def reload_model():
    """热更新模型"""
    global ocr_models
    
    # 卸载旧模型
    for key in list(ocr_models.keys()):
        del ocr_models[key]
    
    # 加载新模型
    custom_model_path = "./models/custom"
    if os.path.exists(custom_model_path):
        ocr_models["general"] = PaddleOCR(
            rec_model_dir=custom_model_path,
            lang='ch'
        )
        return {"status": "success", "message": "Model reloaded"}
    else:
        return {"status": "error", "message": "Custom model not found"}
```

---

## 七、自动训练系统

### 7.1 架构设计

```
┌──────────────────────────────────────────────────────────────┐
│                     自动训练系统                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │ 数据收集器   │────→│ 训练调度器   │────→│ 训练执行器   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│         ↑                   ↓                   ↓            │
│         │            ┌─────────────┐     ┌─────────────┐    │
│         │            │ 模型评估器   │←────│ 模型导出器   │    │
│         │            └──────┬──────┘     └─────────────┘    │
│         │                   ↓                               │
│         │            ┌─────────────┐                        │
│         └────────────│ 模型部署器   │                        │
│                      └─────────────┘                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 自动训练配置

创建 `config/auto_train.yaml`：

```yaml
auto_training:
  enabled: true
  
  # 数据收集
  data_collection:
    enabled: true
    save_dir: ./data/collected
    min_samples: 100       # 触发训练的最小样本数
    
  # 训练调度
  scheduler:
    check_interval: 3600   # 检查间隔（秒）
    train_time: "02:00"    # 训练时间（避免高峰期）
    
  # 训练参数
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.0001
    save_dir: ./output/finetune
    
  # 评估参数
  evaluation:
    test_data: ./data/samples
    min_accuracy: 0.90     # 最低准确率阈值
    
  # 部署参数
  deployment:
    auto_deploy: true      # 自动部署
    backup_old: true       # 备份旧模型
    rollback_on_fail: true # 失败时回滚
```

### 7.3 自动训练脚本

完整的自动训练脚本已实现：[scripts/auto_train.py](file:///c%3A/dev_project/ocr/scripts/auto_train.py)

**使用方式：**

```bash
# 手动触发训练
python scripts/auto_train.py

# 自定义阈值
python scripts/auto_train.py --threshold 500

# 自动部署
python scripts/auto_train.py --auto-deploy
```

### 7.4 定时任务（Cron）

在 Linux 上设置定时训练：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨 2 点检查并训练）
0 2 * * * cd /path/to/CaptchaBreaker && python scripts/auto_train.py >> logs/auto_train.log 2>&1
```

在 Windows 上设置定时任务：

```powershell
# 创建计划任务
$action = New-ScheduledTaskAction -Execute "python" -Argument "scripts/auto_train.py" -WorkingDirectory "C:\dev_project\ocr"
$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM
Register-ScheduledTask -TaskName "OCR Auto Training" -Action $action -Trigger $trigger -User "SYSTEM" -RunLevel Highest
```

---

## 八、增量训练（持续优化）

### 8.1 为什么需要增量训练

随着使用时间的增长，系统会收集到越来越多的验证码图片：

```
第 1 天: 100 张图片
第 7 天: 700 张图片
第 30 天: 3000 张图片
第 90 天: 9000 张图片
```

**增量训练的优势：**
1. **适应新样式**：网站可能更新验证码样式
2. **提高准确率**：更多数据 = 更好的模型
3. **自动化**：无需人工干预

### 8.2 增量训练策略

```python
def incremental_training():
    """增量训练策略"""
    
    # 1. 检查新增数据
    new_samples = count_new_samples()
    
    # 2. 达到训练阈值
    if new_samples >= 500:
        # 3. 使用上次最佳模型继续训练
        model = load_model("./output/finetune/best_accuracy.pdparams")
        
        # 4. 在新数据上训练 10 个 epoch
        model.train(
            new_data="./data/collected",
            epochs=10,
            learning_rate=1e-5  # 使用更小的学习率
        )
        
        # 5. 评估并部署
        accuracy = evaluate(model)
        if accuracy > baseline_accuracy:
            deploy(model)
```

### 8.3 数据管理

随着数据量增加，需要定期清理：

```bash
# 清理超过 30 天的数据
find ./data/collected -type f -mtime +30 -delete

# 或者保留所有数据，但只使用最近 10000 张训练
python scripts/prepare_training.py --max_samples 10000
```

---

## 九、常见问题

### 9.1 训练数据不够怎么办？

**解决方案：**
1. **降低阈值**：`python scripts/auto_train.py --threshold 50`
2. **数据增强**：对现有数据进行旋转、缩放、添加噪声
3. **迁移学习**：使用更大的预训练模型

### 9.2 训练后准确率反而下降？

**可能原因：**
1. 学习率太高 → 降低学习率
2. 数据标注错误 → 检查标注文件
3. 训练数据分布不均 → 增加样本多样性

**解决方案：**
- 使用更小的学习率（1e-5）
- 训练更少 epochs（10-20）
- 使用早停机制

### 9.3 如何确保自动训练的稳定性？

**最佳实践：**
1. **监控训练日志**：`tail -f logs/auto_train.log`
2. **设置告警**：准确率低于阈值时发送通知
3. **保留回滚机制**：自动训练失败时恢复旧模型
4. **定期人工审核**：每周检查一次训练效果

---

## 十、完整训练示例

### 10.1 从零开始训练

```bash
# Step 1: 收集数据（至少 100 张）
# 通过正常使用系统，数据会自动收集到 data/collected/

# Step 2: 准备训练数据
python scripts/prepare_training.py

# Step 3: 训练模型
python scripts/train_model.py

# Step 4: 评估模型
python scripts/evaluate_model.py

# Step 5: 部署模型
python scripts/deploy_model.py

# Step 6: 重启服务
docker-compose restart
```

### 10.2 设置自动训练

```bash
# 1. 配置自动训练参数
cp config/auto_train.yaml.example config/auto_train.yaml

# 2. 启动自动训练调度器
python scripts/auto_train_daemon.py

# 3. 或者设置定时任务
# Linux:
crontab -e
0 2 * * * cd /path/to/CaptchaBreaker && python scripts/auto_train.py

# Windows:
# 使用 Windows Task Scheduler
```

---

## 参考资料

- [PaddleOCR 训练文档](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/training.md)
- [PaddleX 官方文档](https://paddlex.readthedocs.io/)
- [CaptchaBreaker 架构文档](ARCHITECTURE.md)
- [CaptchaBreaker Docker 部署文档](DOCKER.md)
