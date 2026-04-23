# API 参考文档

## 基础信息

| 项目 | 详情 |
|------|------|
| 服务地址 | `http://localhost:8000` |
| 协议 | HTTP/HTTPS |
| 数据格式 | JSON / Multipart Form Data |
| 版本 | 5.0.0 |

---

## 接口列表

### 1. 健康检查

```http
GET /health
```

**响应**:
```json
{
    "status": "healthy",
    "ocr_models_loaded": true
}
```

**状态码**:
| Code | 说明 |
|------|------|
| 200 | 服务正常 |
| 500 | 服务异常 |

---

### 2. 上传图片识别

```http
POST /ocr/upload
Content-Type: multipart/form-data
```

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| file | File | 是 | - | 图片文件 (PNG/JPG/BMP/WEBP) |
| language | String | 否 | `general` | 识别语言: `general` (中文) / `en` (英文) |

**请求示例** (cURL):
```bash
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@captcha.png" \
  -F "language=en"
```

**请求示例** (Python):
```python
import requests

with open("captcha.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/upload",
        files={"file": ("captcha.png", f, "image/png")},
        data={"language": "en"}
    )
    print(response.json())
```

**响应示例** (200 OK):
```json
{
    "success": true,
    "image_id": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4",
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
        "Upscale from 100x40 to 200x80",
        "Light contrast enhancement"
    ]
}
```

**错误响应** (400 Bad Request):
```json
{
    "detail": "File must be an image"
}
```

**错误响应** (500 Internal Server Error):
```json
{
    "detail": "OCR processing failed: ..."
}
```

**状态码**:
| Code | 说明 |
|------|------|
| 200 | 识别成功 |
| 400 | 请求参数错误 (非图片文件/不支持的格式) |
| 500 | 服务器内部错误 |

---

### 3. Base64 图片识别

```http
POST /ocr/base64
Content-Type: application/x-www-form-urlencoded
```

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| image_data | String | 是 | - | Base64 编码图片 (可带或不带 data URI 前缀) |
| language | String | 否 | `general` | 识别语言 |

**请求示例** (cURL):
```bash
curl -X POST "http://localhost:8000/ocr/base64" \
  -d "image_data=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..." \
  -d "language=en"
```

**请求示例** (Python):
```python
import requests
import base64

with open("captcha.png", "rb") as f:
    b64_data = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/ocr/base64",
    data={
        "image_data": f"data:image/png;base64,{b64_data}",
        "language": "en"
    }
)
print(response.json())
```

**响应格式**: 同 `/ocr/upload`

---

### 4. URL 图片识别

```http
POST /ocr/url
Content-Type: application/json
```

**参数**:
| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| image_url | String | 是 | - | 图片的完整 URL |
| language | String | 否 | `general` | 识别语言 |

**请求示例** (cURL):
```bash
curl -X POST "http://localhost:8000/ocr/url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/captcha.png",
    "language": "en"
  }'
```

**请求示例** (Python):
```python
import requests

response = requests.post(
    "http://localhost:8000/ocr/url",
    json={
        "image_url": "https://example.com/captcha.png",
        "language": "en"
    }
)
print(response.json())
```

**响应格式**: 同 `/ocr/upload`

---

## 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `success` | Boolean | 是否识别成功 |
| `image_id` | String | 本次请求的唯一标识 (UUID hex) |
| `texts` | Array | 识别到的文本列表 |
| `texts[].text` | String | 识别到的文本内容 |
| `texts[].confidence` | Float | 识别置信度 (0.0 - 1.0) |
| `texts[].bounding_box` | Object | 文本在图片中的边界框 |
| `texts[].bounding_box.x_min` | Float | 左边界 x 坐标 |
| `texts[].bounding_box.y_min` | Float | 上边界 y 坐标 |
| `texts[].bounding_box.x_max` | Float | 右边界 x 坐标 |
| `texts[].bounding_box.y_max` | Float | 下边界 y 坐标 |
| `full_text` | String | 所有识别文本合并后的结果 (无空格) |
| `language` | String | 使用的识别语言 |
| `message` | String | 处理结果描述 |
| `preprocessing_applied` | Array | 应用的预处理步骤列表 |

---

## 支持的文件格式

| 格式 | MIME Type | 文件扩展名 |
|------|-----------|-----------|
| PNG | `image/png` | `.png` |
| JPEG | `image/jpeg` | `.jpg`, `.jpeg` |
| BMP | `image/bmp` | `.bmp` |
| WebP | `image/webp` | `.webp` |

---

## 错误码说明

| HTTP Code | 说明 | 解决建议 |
|-----------|------|----------|
| 400 | 请求参数错误 | 检查文件格式和参数 |
| 500 | 服务器内部错误 | 检查日志，可能是模型或预处理问题 |
| 503 | 服务不可用 | 等待服务启动完成 |

---

## 使用建议

### 1. 语言选择

| 场景 | 推荐 language | 说明 |
|------|---------------|------|
| 英文字母+数字验证码 | `en` | 英文模型，针对字母数字优化 |
| 含中文的验证码 | `general` | 中文模型，支持中英文混合 |
| 不确定 | `en` | 大多数验证码为英文 |

### 2. 性能优化

| 建议 | 说明 |
|------|------|
| 复用连接 | 使用 `requests.Session()` 复用 HTTP 连接 |
| 批量处理 | 并发请求多个图片，但注意服务器资源 |
| 超时设置 | 建议设置 30s 超时，避免长时间等待 |

### 3. 错误处理

```python
import requests

try:
    response = requests.post(
        "http://localhost:8000/ocr/upload",
        files={"file": open("captcha.png", "rb")},
        data={"language": "en"},
        timeout=30
    )
    response.raise_for_status()
    result = response.json()
    if result["success"]:
        print(f"识别结果: {result['full_text']}")
    else:
        print(f"识别失败: {result['message']}")
except requests.exceptions.Timeout:
    print("请求超时")
except requests.exceptions.RequestException as e:
    print(f"请求错误: {e}")
```

---

## 交互式文档

启动服务后，可通过浏览器访问以下地址查看交互式 API 文档：

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
