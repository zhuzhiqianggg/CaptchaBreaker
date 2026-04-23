# 优化路线图

## 目录
- [1. 当前状态](#1-当前状态)
- [2. 失败模式分析](#2-失败模式分析)
- [3. 优化方案](#3-优化方案)
- [4. 实施步骤](#4-实施步骤)
- [5. 预期效果](#5-预期效果)

---

## 1. 当前状态

### 1.1 性能指标 (v5.0)

| 指标 | 数值 | 说明 |
|------|------|------|
| 真实准确率 | ~60% | 以文件名为基准的完全匹配 |
| 样本总数 | 31 张 | 混合类型的验证码 |
| 平均置信度 | ~0.85 | OCR 模型自身评估 |
| 平均延迟 | ~3.5s | CPU 模式 |

### 1.2 版本演进

| 版本 | 预处理策略 | 准确率 | 说明 |
|------|-----------|--------|------|
| v1.0-v3.0 | 实验阶段 | - | API 调试和模型适配 |
| v4.0 | 过度预处理 | 0% | 二值化+连通域破坏了文字 |
| v5.0 | 轻量预处理 | ~60% | 上采样+对比度增强 |

---

## 2. 失败模式分析

### 2.1 典型失败案例

| 图片 | 期望 | 识别 | 失败原因 |
|------|------|------|----------|
| 6QxJ.png | 6qxj | qxj | **首字符丢失**: "6" 未被检测到 |
| 78sq.png | 78sq | bs8l | **字符混淆**: 7→b, q→8 |
| aZWA.png | azwa | zwa | **首字符丢失**: "a" 未被检测到 |

### 2.2 失败原因分类

| 失败类型 | 占比 | 原因 | 优化方向 |
|----------|------|------|----------|
| 首字符丢失 | ~30% | 检测器边界计算错误 | 调整检测参数 |
| 字符混淆 | ~40% | 字体与训练数据差异 | 模型 fine-tune |
| 额外字符 | ~15% | 噪点被误识别 | 预处理优化 |
| 完全错误 | ~15% | 多种因素叠加 | 综合优化 |

### 2.3 验证码对抗特性

验证码设计中用于对抗 OCR 的技术：

1. **字符粘连**: 字符间距极小，导致检测器合并或拆分错误
2. **干扰线条**: 穿过字符的横线/曲线，导致字符被分割
3. **非常规字体**: 非标准字体，与训练数据分布差异大
4. **小尺寸**: ~100x40 像素，字符信息量有限
5. **颜色干扰**: 背景与字符颜色相近，增加分割难度

---

## 3. 优化方案

### 3.1 方案一：模型 Fine-tuning (推荐)

**预期准确率**: 90-95%  
**工作量**: 中等 (1-2天)  
**难度**: 中等

#### 原理
使用目标验证码类型的大量样本，对 PaddleOCR 的识别模型进行微调训练，使其适应该类验证码的字体、风格和干扰特征。

#### 实施步骤

```
Step 1: 数据收集 (3-4小时)
├─ 收集 1000+ 张同类验证码
├─ 以文件名为 Ground Truth
└─ 按 80/20 分为训练集和验证集

Step 2: 数据标注 (1-2小时)
├─ 生成 PaddleOCR 训练格式数据
├─ 创建 rec_gt.txt 标注文件
└─ 整理训练图片到统一目录

Step 3: 模型训练 (2-4小时)
├─ 基于 en_PP-OCRv5_mobile_rec 预训练模型
├─ 修改配置文件
├─ 执行训练命令
└─ 监控训练过程

Step 4: 模型评估与部署 (1小时)
├─ 在验证集上测试准确率
├─ 导出 inference 模型
└─ 替换项目中的默认模型
```

#### 训练命令示例

```bash
# 单卡训练
python tools/train.py \
    -c configs/rec/rec_mv3_en_PP-OCRv5_train.yml \
    -o Global.pretrained_model=./pretrain_models/en_PP-OCRv5_mobile_rec/best_accuracy \
    -o Global.train.batch_size_per_card=256 \
    -o Global.train.epoch_num=100

# 评估
python tools/eval.py \
    -c configs/rec/rec_mv3_en_PP-OCRv5_train.yml \
    -o Global.checkpoints=./output/rec/best_accuracy
```

#### 关键参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| batch_size | 256 | 根据显存调整 |
| learning_rate | 0.0005 | 微调学习率 |
| epoch | 50-100 | 训练轮数 |
| optimizer | Adam | 优化器 |

### 3.2 方案二：多策略集成

**预期准确率**: 75-85%  
**工作量**: 低 (0.5天)  
**难度**: 低

#### 原理
使用多种不同的预处理策略对同一张图片进行多次识别，然后通过投票或评分选择最优结果。

#### 实施架构

```python
def multi_strategy_recognition(image):
    strategies = [
        strategy_v1(image),  # 原始 + 上采样
        strategy_v2(image),  # 灰度 + 对比度增强
        strategy_v3(image),  # 自适应阈值 + 去噪
        strategy_v4(image),  # 边缘增强
    ]
    
    results = []
    for processed_img in strategies:
        result = ocr_model.ocr(processed_img)
        results.append(result)
    
    # 投票选出最优结果
    return vote_for_best(results)
```

#### 候选策略

| 策略 | 预处理方法 | 适用场景 |
|------|-----------|----------|
| v1 | 原始图 + 3x上采样 | 标准验证码 |
| v2 | 灰度 + 对比度2.0x + 上采样 | 低对比度验证码 |
| v3 | 自适应阈值 + 去噪 | 高噪点验证码 |
| v4 | 边缘增强 + 上采样 | 模糊验证码 |

#### 投票算法

```python
def vote_for_best(results):
    scores = []
    for result in results:
        # 综合评分 = 平均置信度 * 0.4 + 文本长度评分 * 0.6
        avg_conf = mean(confidences)
        len_score = min(len(text) / 4, 1.0)  # 假设4个字符
        score = avg_conf * 0.4 + len_score * 0.6
        scores.append(score)
    
    best_idx = argmax(scores)
    return results[best_idx]
```

### 3.3 方案三：针对性预处理优化

**预期准确率**: 70-80%  
**工作量**: 低 (0.5天)  
**难度**: 低

#### 原理
根据验证码的具体特征，设计针对性的预处理算法。

#### 候选优化

| 优化项 | 方法 | 预期效果 |
|--------|------|----------|
| 自适应阈值 | 根据图片亮度选择 Otsu 或固定阈值 | +5-10% |
| 干扰线去除 | 霍夫变换检测并去除直线干扰 | +5-8% |
| 字符分割增强 | 形态学操作分离粘连字符 | +3-5% |
| 多尺度上采样 | 尝试 2x/3x/4x 不同放大倍数 | +3-5% |

### 3.4 方案四：后处理纠错

**预期准确率**: +5-10%  
**工作量**: 极低 (1小时)  
**难度**: 极低

#### 原理
基于验证码的常见混淆模式，对 OCR 结果进行后处理修正。

#### 实现示例

```python
def post_process(text):
    # 常见混淆字符替换
    corrections = {
        '0': 'O', '1': 'l', '5': 'S',
        '7': 'T', '8': 'B', '9': 'g'
    }
    # 基于验证码特征的纠错逻辑
    return corrected_text
```

### 3.5 方案五：GPU 加速

**预期准确率**: 不变  
**预期延迟**: 3.5s → 0.3-0.7s  
**工作量**: 低 (需安装 GPU 驱动和 CUDA)

#### 实施步骤

```bash
# 安装 GPU 版本 PaddlePaddle
pip install paddlepaddle-gpu==3.0.0

# 代码中指定 GPU
PaddleOCR(lang='en', use_gpu=True)
```

---

## 4. 实施步骤

### 4.1 推荐实施路径

```
Phase 1: 快速提升 (1-2天)
├─ 方案四: 后处理纠错 (+5-10%)
├─ 方案二: 多策略集成 (+10-15%)
└─ 方案三: 针对性预处理 (+5-10%)
│   └─ 预期: 70-80%

Phase 2: 深度优化 (3-5天)
├─ 方案一: 模型 Fine-tune (+20-30%)
│   ├─ 数据收集
│   ├─ 训练准备
│   ├─ 模型训练
│   └─ 模型评估与部署
└─ 预期: 90-95%

Phase 3: 性能优化 (1天)
├─ 方案五: GPU 加速
│   ├─ 安装 CUDA + cuDNN
│   ├─ 安装 paddlepaddle-gpu
│   └─ 性能测试
└─ 预期延迟: 0.3-0.7s
```

### 4.2 数据收集指南

1. **收集来源**: 目标网站验证码接口
2. **数量要求**: 至少 1000 张，推荐 5000+
3. **命名规范**: 以验证码内容为文件名 (如 `A7k9.png`)
4. **数据检查**: 人工抽样验证文件名与内容一致性
5. **数据划分**:
   - 训练集: 80% (800+ 张)
   - 验证集: 20% (200+ 张)
   - 测试集: 新收集 (100+ 张)

### 4.3 Fine-tune 数据准备

```
data/
├── captcha_finetune/
│   ├── train/
│   │   ├── A7k9.png
│   │   ├── BRTf.png
│   │   └── ... (800+ images)
│   ├── val/
│   │   ├── 2zrw.png
│   │   ├── 6QxJ.png
│   │   └── ... (200+ images)
│   └── rec_gt_train.txt
│       A7k9.png    A7k9
│       BRTf.png    BRTf
```

---

## 5. 预期效果

### 5.1 各阶段预期准确率

| 阶段 | 方案 | 预期准确率 | 累计准确率 |
|------|------|-----------|-----------|
| 基线 (v5.0) | - | 60% | 60% |
| Phase 1.1 | 后处理纠错 | +5-10% | 65-70% |
| Phase 1.2 | 多策略集成 | +10-15% | 70-80% |
| Phase 1.3 | 针对性预处理 | +5-10% | 75-85% |
| Phase 2 | 模型 Fine-tune | +15-25% | **90-95%** |

### 5.2 风险评估

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数据量不足 | Fine-tune 效果差 | 增加数据收集量 |
| 过拟合 | 新验证码识别率低 | 增加验证集，early stopping |
| 训练时间长 | 项目延期 | 使用 GPU 加速训练 |
| 验证码规则变更 | 模型失效 | 定期更新训练数据 |

---

## 6. 参考资料

### 6.1 PaddleOCR 官方文档
- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Fine-tune 指南](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/finetune.md)
- [模型训练](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/training.md)

### 6.2 相关论文
- PP-OCRv5: Latest OCR System from Baidu
- DB: Differentiable Binarization for Text Detection
- CRNN: Convolutional Recurrent Neural Network

### 6.3 工具
- [PaddleX](https://github.com/PaddlePaddle/PaddleX) - 可视化训练工具
- [Label Studio](https://labelstud.io/) - 数据标注工具
