# OCR 验证码识别系统 - 技术文档

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 技术架构](#2-技术架构)
- [3. 项目结构](#3-项目结构)
- [4. 核心技术实现](#4-核心技术实现)
- [5. 准确率优化策略](#5-准确率优化策略)
- [6. API 接口文档](#6-api-接口文档)
- [7. 性能测试与评估](#7-性能测试与评估)
- [8. 部署与使用](#8-部署与使用)
- [9. 局限性与改进方向](#9-局限性与改进方向)

---

## 1. 项目概述

### 1.1 项目背景
本项目旨在构建一个**本地部署**的 OCR 验证码识别服务，使用 PaddleOCR 深度学习模型结合图像预处理技术，实现自动化验证码识别。

### 1.2 核心功能
- **图片上传识别**：支持 PNG/JPG/BMP/WEBP 格式
- **Base64 编码识别**：直接传输 Base64 编码图片数据
- **URL 识别**：通过图片 URL 远程识别
- **智能图像预处理**：自动降噪、二值化、连通域分析
- **JSON 格式返回**：结构化返回识别结果、置信度、边界框

### 1.3 技术选型
| 组件 | 技术 | 版本 |
|------|------|------|
| OCR 引擎 | PaddleOCR (PP-OCRv5) | 3.4.1 |
| 深度学习框架 | PaddlePaddle | 3.0.0 |
| Web 框架 | FastAPI | 0.115.0 |
| 图像处理 | Pillow + OpenCV | 10.4.0 + 4.10.0 |
| 数值计算 | NumPy | 1.26.4 |

---

## 2. 技术架构

### 2.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         客户端请求                                │
│                    (Upload / Base64 / URL)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Web Server                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  /ocr/upload │  │ /ocr/base64  │  │    /ocr/url          │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         └─────────────────┴─────────────────────┘               │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              图像预处理管道 (Preprocessing Pipeline)         │ │
│  │  ┌────────────┐  ┌───────────┐  ┌────────────────────┐   │ │
│  │  │ 灰度化+增强 │→│ Otsu二值化 │→│ 形态学操作(开/闭)   │   │ │
│  │  └────────────┘  └───────────┘  └────────────────────┘   │ │
│  │         ↓                                                  │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │ 连通域分析 (Connected Components Analysis)         │    │ │
│  │  │ - 面积过滤 (min_area=15)                           │    │ │
│  │  │ - 噪点去除                                         │    │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  │         ↓                                                  │ │
│  │  ┌──────────────────────────────────────────────────┐    │ │
│  │  │ 3x 上采样 (Lanczos 插值)                           │    │ │
│  │  └──────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   PaddleOCR 引擎                             │ │
│  │  ┌──────────────────┐  ┌──────────────────┐               │ │
│  │  │ PP-OCRv5_server_ │→│ PP-OCRv5_server_  │               │ │
│  │  │ det (检测器)      │  │ rec (识别器)       │               │ │
│  │  └──────────────────┘  └──────────────────┘               │ │
│  └────────────────────────────────────────────────────────────┘ │
│                             │                                    │
│                             ▼                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    结果解析与输出                             │ │
│  │  - 文本提取与去空格                                          │ │
│  │  - 置信度计算                                                │ │
│  │  - 边界框坐标提取                                            │ │
│  │  - JSON 结构化返回                                           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 数据流

```
原始图片 → RGB转换 → 灰度化 → 对比度增强 → Otsu二值化 
       → 形态学操作 → 连通域过滤 → 3x上采样 → OCR识别 
       → 结果解析 → JSON响应
```

---

## 3. 项目结构

```
c:\dev_project\ocr\
├── main.py                          # 主程序 (FastAPI + PaddleOCR)
├── requirements.txt                 # Python 依赖清单
├── temp/                            # 临时文件目录 (自动清理)
├── code_images/                     # 测试验证码图片集
│   ├── 2zrw.png                     # 验证码: 2zrw
│   ├── 6QxJ.png                     # 验证码: 6QxJ
│   ├── 78sq.png                     # 验证码: 78sq
│   ├── ... (共31张图片)
│   └── crop_test.png
├── test_images/                     # 自动生成的测试图片
├── test_real_accuracy.py            # 真实准确率测试脚本
├── test_code_images.py              # 基础测试脚本
├── test_api.py                      # API 接口测试
├── test_ocr.py                      # 单张图片测试
├── debug_ocr.py                     # OCR 输出调试
├── test_report.py                   # 报告生成脚本
├── ocr_test_results.json            # 原始测试结果
├── ocr_test_results_optimized.json  # 优化版测试结果
└── README.md                        # 本技术文档
```

---

## 4. 核心技术实现

### 4.1 图像预处理管道

图像预处理是提升验证码识别准确率的关键环节。验证码图片通常包含以下干扰因素：
- **背景噪声**：随机噪点、斑点
- **干扰线条**：横线、竖线、曲线
- **颜色干扰**：彩色背景、渐变
- **字符变形**：旋转、扭曲、粘连

我们的预处理管道针对这些干扰进行了专门设计：

#### 4.1.1 灰度化 + 对比度增强

```python
# 转换为灰度图
img_gray = image.convert("L")

# 对比度增强 (2.0x)
enhancer = ImageEnhance.Contrast(img_gray)
img_gray = enhancer.enhance(2.0)
```

**原理**：将彩色图片转为灰度图，消除颜色干扰。增强对比度使字符和背景的差异更明显，有利于后续二值化。

#### 4.1.2 Otsu 二值化

```python
import cv2
import numpy as np

img_array = np.array(img_gray)
_, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

**原理**：Otsu 方法自动寻找最佳阈值，将灰度图转为纯黑白二值图。相比固定阈值，Otsu 能自适应不同亮度的图片。

#### 4.1.3 形态学操作

```python
kernel = np.ones((2, 2), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 闭运算
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)   # 开运算
```

**原理**：
- **闭运算 (Close)**：先膨胀后腐蚀，填充字符内部的小空洞，连接断裂的笔画
- **开运算 (Open)**：先腐蚀后膨胀，去除小噪点和干扰线条

#### 4.1.4 连通域分析 + 面积过滤

```python
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

min_area = 15  # 最小连通域面积
cleaned = np.zeros_like(binary)

for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        cleaned[labels == i] = 0  # 保留大字符合并区域
    else:
        cleaned[labels == i] = 255  # 过滤小噪点
```

**原理**：
- 使用连通域分析找出所有独立的白色区域
- 每个字符通常由多个像素组成，面积较大
- 噪点通常只有几个像素，面积很小
- 通过设置最小面积阈值 (min_area=15)，有效去除噪点

#### 4.1.5 3x 上采样

```python
scale_factor = 3
new_width = original_width * scale_factor
new_height = original_height * scale_factor
cleaned_pil = cleaned_pil.resize((new_width, new_height), Image.LANCZOS)
```

**原理**：验证码图片通常很小（~100x40像素），直接输入 OCR 模型时字符像素太少，难以识别。3x 上采样后字符更清晰，识别率大幅提升。

### 4.2 PaddleOCR 模型

#### 4.2.1 模型架构

PaddleOCR 使用两阶段架构：

```
预处理图片 → 文本检测 (Detector) → 文本识别 (Recognizer) → 输出文本
```

**文本检测器 (PP-OCRv5_server_det)**：
- 基于 DB (Differentiable Binarization) 算法
- 输出文本区域的边界框坐标
- 模型大小：87.9 MB

**文本识别器 (en_PP-OCRv5_mobile_rec)**：
- 基于 CRNN (Convolutional Recurrent Neural Network)
- 支持英文字母、数字、中文字符
- 模型大小：移动端优化版，约 10 MB

#### 4.2.2 模型文件位置

```
C:\Users\zhuzhiqiang\.paddlex\official_models\
├── PP-LCNet_x1_0_doc_ori/          # 文档方向分类器
├── UVDoc/                           # 文档矫正模型
├── PP-LCNet_x1_0_textline_ori/      # 文本行方向分类器
├── PP-OCRv5_server_det/             # 文本检测器
├── PP-OCRv5_server_rec/             # 中文识别器
└── en_PP-OCRv5_mobile_rec/          # 英文识别器
```

### 4.3 结果解析

```python
def parse_ocr_result(result) -> Tuple[List[OCRResult], str]:
    # 提取识别文本、置信度、边界框
    for i, recognized_text in enumerate(rec_texts):
        confidence = float(rec_scores[i])
        bbox = extract_bounding_box(rec_polys[i])
        texts.append(OCRResult(text=recognized_text, confidence=confidence, bbox=bbox))
    
    # 合并文本，去除空格
    full_text = "".join(full_text_parts)
    full_text = re.sub(r'\s+', '', full_text)
    return texts, full_text
```

---

## 5. 准确率优化策略

### 5.1 当前优化效果

| 阶段 | 平均置信度 | 说明 |
|------|-----------|------|
| 原始图片 (无预处理) | ~0.9364 | 简单识别，但字符丢失严重 |
| 基础预处理 | ~0.7303 | 过度处理导致部分字符变形 |
| 当前版本 (v4.0) | 待测试 | 平衡的预处理策略 |

### 5.2 为什么难以达到 100% 准确率？

验证码设计的初衷就是**对抗自动识别**，以下因素使得 100% 准确率极具挑战：

1. **字符粘连**：如 "6QxJ" 中字符间距极小，OCR 容易合并识别
2. **字符旋转**：部分验证码字符有微小旋转，影响识别
3. **干扰线条**：横线穿过字符，可能导致字符被分割
4. **字体变形**：验证码使用非常规字体，与 OCR 训练数据差异大
5. **小尺寸**：原始图片仅 ~100x40，信息量有限

### 5.3 进一步提升准确率的策略

#### 策略 1：针对性预处理调优

```python
# 针对不同验证码类型使用不同参数
def adaptive_preprocess(image):
    if image.width < 100:
        scale = 4  # 更小的图片需要更大放大倍数
    else:
        scale = 3
    
    if is_high_contrast(image):
        threshold_method = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    else:
        threshold_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
```

#### 策略 2：多策略投票机制

```python
def multi_strategy_vote(image):
    results = []
    for strategy in [strategy_v1, strategy_v2, strategy_v3]:
        result = ocr(strategy(image))
        results.append(result)
    
    # 取最多票数的结果
    return most_common(results)
```

#### 策略 3：后处理纠错

```python
def post_process_correction(text):
    # 常见混淆字符替换
    corrections = {
        '0': 'O', '1': 'l', '5': 'S',
        '7': 'T', '8': 'B', '9': 'g'
    }
    # 根据验证码特征进行针对性修正
    return corrected_text
```

#### 策略 4：专用验证码识别模型

如果验证码类型相对固定，可以：
1. 收集大量同类验证码样本
2. 使用 PaddleOCR 的训练功能 fine-tune 模型
3. 专门训练针对该类验证码的识别器

**预期效果**：fine-tune 后可达 90%+ 准确率

---

## 6. API 接口文档

### 6.1 基础信息

- **服务地址**：`http://localhost:8000`
- **接口协议**：HTTP/HTTPS
- **数据格式**：JSON / Multipart Form Data

### 6.2 接口列表

#### 6.2.1 健康检查

```
GET /health
```

**响应**：
```json
{
    "status": "healthy",
    "ocr_models_loaded": true
}
```

#### 6.2.2 上传图片识别

```
POST /ocr/upload
Content-Type: multipart/form-data
```

**参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file | File | 是 | 图片文件 (PNG/JPG/BMP/WEBP) |
| language | String | 否 | 语言: "general"(中文) / "en"(英文)，默认 "general" |

**示例** (cURL)：
```bash
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@captcha.png" \
  -F "language=en"
```

**示例** (Python)：
```python
import requests

with open("captcha.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/upload",
        files={"file": f},
        data={"language": "en"}
    )
    print(response.json())
```

#### 6.2.3 Base64 图片识别

```
POST /ocr/base64
Content-Type: application/x-www-form-urlencoded
```

**参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| image_data | String | 是 | Base64 编码图片 (可带 data URI 前缀) |
| language | String | 否 | 语言，默认 "general" |

**示例** (Python)：
```python
import base64
import requests

with open("captcha.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/ocr/base64",
    data={"image_data": b64, "language": "en"}
)
```

#### 6.2.4 URL 图片识别

```
POST /ocr/url
Content-Type: application/json
```

**请求体**：
```json
{
    "image_url": "https://example.com/captcha.png",
    "language": "en"
}
```

### 6.3 统一响应格式

```json
{
    "success": true,
    "image_id": "a1b2c3d4e5f6...",
    "texts": [
        {
            "text": "A7k9",
            "confidence": 0.9956,
            "bounding_box": {
                "x_min": 75.0,
                "y_min": 17.0,
                "x_max": 200.0,
                "y_max": 69.0
            }
        }
    ],
    "full_text": "A7k9",
    "language": "en",
    "message": "Successfully recognized 1 text(s)",
    "preprocessing_applied": [
        "Original size: 100x40",
        "Contrast enhancement (2.0x)",
        "Brightness enhancement (1.0x)",
        "Otsu thresholding",
        "Morphological operations (close + open)",
        "Connected components filtering (min_area=15)",
        "Upscale 3x to 300x120"
    ]
}
```

---

## 7. 性能测试与评估

### 7.1 测试环境

| 项目 | 规格 |
|------|------|
| CPU | Intel (未配备GPU) |
| Python | 3.13.1 |
| 测试图片 | 31 张验证码 |
| 图片尺寸 | ~100x40 像素 |

### 7.2 测试方法

以**图片文件名**作为正确答案（Ground Truth），计算真实识别准确率。

**匹配规则**：
1. **完全匹配**：识别结果与文件名完全一致（忽略大小写和空格）
2. **部分匹配**：识别结果包含大部分文件名字符
3. **完全不匹配**：识别结果与文件名差异较大

### 7.3 测试指标

| 指标 | 定义 |
|------|------|
| 准确率 (Accuracy) | 完全正确的图片数 / 总图片数 |
| 平均置信度 | OCR 模型输出的平均置信度 |
| 平均处理时间 | 单张图片从上传到返回的平均时间 |

### 7.4 运行测试

```bash
# 运行真实准确率测试
python test_real_accuracy.py

# 运行基础测试
python test_code_images.py

# 运行 API 测试
python test_api.py
```

---

## 8. 部署与使用

### 8.1 环境要求

- Python 3.10+
- Windows / Linux / macOS
- 内存 >= 4GB
- 磁盘空间 >= 2GB (模型文件约 500MB)

### 8.2 安装步骤

```bash
# 1. 进入项目目录
cd c:\dev_project\ocr

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python main.py
```

### 8.3 服务启动后

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

服务启动后访问：
- API 文档：http://localhost:8000/
- 健康检查：http://localhost:8000/health

### 8.4 生产部署建议

```bash
# 使用 gunicorn 生产级部署
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

---

## 9. 局限性与改进方向

### 9.1 当前局限性

1. **通用 OCR 模型的局限**
   - PaddleOCR 是通用 OCR 模型，针对文档、场景文字训练
   - 验证码对抗性设计超出通用 OCR 的优化范围

2. **预处理参数的固定性**
   - 当前预处理参数是针对某类验证码调优的
   - 不同类型的验证码可能需要不同的参数

3. **无反馈学习机制**
   - 当前系统无法从识别错误中学习
   - 缺少自动纠错和自适应能力

### 9.2 改进方向

#### 方向 1：Fine-tune 专用模型 (推荐)
```
预期准确率：90%+
工作量：中等
步骤：
1. 收集 1000+ 同类验证码样本
2. 标注正确答案
3. 使用 PaddleOCR 训练脚本 fine-tune
4. 替换默认模型
```

#### 方向 2：验证码特征分析 + 专用预处理
```
预期准确率：70-80%
工作量：低
步骤：
1. 分析验证码干扰特征
2. 针对性设计预处理算法
3. 调整 OCR 参数
```

#### 方向 3：多模型集成
```
预期准确率：80-85%
工作量：中等
步骤：
1. 同时使用多个 OCR 模型
2. 投票选出最可能正确的结果
3. 结合规则进行后处理
```

### 9.3 联系方式与反馈

如有问题或建议，请查看项目目录下的测试脚本了解详细实现。

---

## 附录

### A. PaddleOCR 模型下载路径
```
C:\Users\zhuzhiqiang\.paddlex\official_models\
```

### B. 常见问题

**Q: 识别速度慢？**
A: CPU 模式下首次加载模型约需 30 秒，后续每张图片约 3-4 秒。

**Q: 如何更换语言？**
A: 请求时指定 `language` 参数：`"en"` 为英文，`"general"` 为中文。

**Q: 识别结果为空？**
A: 可能原因：图片太小、干扰太强。尝试增大图片或调整预处理参数。
