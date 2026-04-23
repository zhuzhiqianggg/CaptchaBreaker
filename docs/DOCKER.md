# Docker 部署指南

## 目录
- [1. 环境要求](#1-环境要求)
- [2. 快速部署](#2-快速部署)
- [3. 详细部署步骤](#3-详细部署步骤)
- [4. 使用指南](#4-使用指南)
- [5. 配置说明](#5-配置说明)
- [6. 常见问题](#6-常见问题)
- [7. 生产环境优化](#7-生产环境优化)

---

## 1. 环境要求

### 最低配置
- **CPU**: 2 核
- **内存**: 4GB
- **磁盘**: 10GB 可用空间
- **系统**: Linux (Ubuntu 20.04+) / Windows 10+ / macOS 10.15+

### 推荐配置
- **CPU**: 4 核+
- **内存**: 8GB+
- **磁盘**: 20GB SSD
- **GPU** (可选): NVIDIA GPU with CUDA 支持 (可提升 5-10x 性能)

### 必需软件
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

---

## 2. 快速部署 (3 步完成)

### 步骤 1: 克隆项目

```bash
git clone https://github.com/zhuzhiqianggg/CaptchaBreaker.git
cd CaptchaBreaker
```

### 步骤 2: 启动服务

```bash
docker-compose up -d
```

### 步骤 3: 验证部署

```bash
curl http://localhost:8000/health
```

返回以下 JSON 表示成功：

```json
{
  "status": "healthy",
  "message": "OCR service is running"
}
```

**总耗时**: ~15-20 分钟 (首次需要下载模型)

---

## 3. 详细部署步骤

### 3.1 安装 Docker

#### Ubuntu/Debian

```bash
# 安装 Docker
curl -fsSL https://get.docker.com | sh

# 启动 Docker 服务
sudo systemctl start docker
sudo systemctl enable docker

# 添加当前用户到 docker 组 (避免每次使用 sudo)
sudo usermod -aG docker $USER
newgrp docker

# 验证安装
docker --version
docker-compose --version
```

#### Windows

1. 下载并安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
2. 启动 Docker Desktop
3. 在 PowerShell 中验证：

```powershell
docker --version
docker-compose --version
```

#### macOS

```bash
# 使用 Homebrew 安装
brew install --cask docker

# 或者从官网下载 Docker Desktop for Mac
```

### 3.2 克隆项目

```bash
git clone https://github.com/zhuzhiqianggg/CaptchaBreaker.git
cd CaptchaBreaker
```

### 3.3 检查配置文件

确保以下文件存在：
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`
- `requirements.txt`
- `app/` 目录

### 3.4 构建并启动

```bash
# 方式 1: 使用 docker-compose (推荐)
docker-compose up -d

# 方式 2: 使用 docker 命令 (高级用户)
docker build -t captcha-breaker:7.0.0 .
docker run -d \
  --name captcha-breaker \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  --restart unless-stopped \
  captcha-breaker:7.0.0
```

### 3.5 查看日志

```bash
# 查看服务状态
docker-compose ps

# 查看启动日志
docker-compose logs -f ocr-api

# 查看最近 100 行日志
docker-compose logs --tail=100 ocr-api
```

### 3.6 验证服务

```bash
# 健康检查
curl http://localhost:8000/health

# 测试 OCR 识别
curl -X POST http://localhost:8000/ocr/upload \
  -F "file=@data/samples/2zrw.png" \
  -F "language=general"
```

---

## 4. 使用指南

### 4.1 基本操作

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down

# 重启服务
docker-compose restart

# 查看日志
docker-compose logs -f

# 更新服务
docker-compose pull
docker-compose up -d --build

# 完全清理
docker-compose down -v --rmi all
```

### 4.2 API 调用示例

#### Python 调用

```python
import requests

# 文件上传方式
with open('captcha.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/ocr/upload',
        files={'file': f},
        data={'language': 'general'}
    )
    print(response.json())

# Base64 方式
import base64

with open('captcha.png', 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode()

response = requests.post(
    'http://localhost:8000/ocr/base64',
    json={
        'image_data': image_base64,
        'language': 'general'
    }
)
print(response.json())

# URL 方式
response = requests.post(
    'http://localhost:8000/ocr/url',
    json={
        'image_url': 'https://example.com/captcha.png',
        'language': 'general'
    }
)
print(response.json())
```

#### cURL 调用

```bash
# 文件上传
curl -X POST http://localhost:8000/ocr/upload \
  -F "file=@captcha.png" \
  -F "language=general"

# Base64
curl -X POST http://localhost:8000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"image_data": "base64_string_here", "language": "general"}'

# URL
curl -X POST http://localhost:8000/ocr/url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/captcha.png", "language": "general"}'
```

#### JavaScript 调用

```javascript
// 使用 fetch API
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('language', 'general');

fetch('http://localhost:8000/ocr/upload', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
```

### 4.3 批量测试

```bash
# 进入容器
docker exec -it captcha-breaker bash

# 运行测试脚本
cd /app
python scripts/test_accuracy.py

# 退出容器
exit
```

---

## 5. 配置说明

### 5.1 环境变量

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `TZ` | Asia/Shanghai | 时区设置 |
| `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK` | True | 禁用模型来源检查 |

### 5.2 docker-compose.yml 配置

```yaml
services:
  ocr-api:
    ports:
      - "8000:8000"  # 宿主机端口:容器端口
    volumes:
      - ./data:/app/data        # 数据目录挂载
      - ./output:/app/output    # 输出目录挂载
      - ./app:/app/app         # 代码热重载 (开发模式)
    deploy:
      resources:
        limits:
          memory: 4G           # 内存上限
          cpus: '2.0'          # CPU 上限
        reservations:
          memory: 2G           # 内存预留
```

### 5.3 修改端口

如果 8000 端口被占用，修改 `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"  # 将宿主机端口改为 8080
```

然后重启服务：

```bash
docker-compose down
docker-compose up -d
```

---

## 6. 常见问题

### 6.1 构建失败

**问题**: Docker 构建过程中报错

**解决方案**:

```bash
# 清理 Docker 缓存
docker system prune -a

# 重新构建 (不使用缓存)
docker-compose build --no-cache

# 查看完整构建日志
docker-compose build --progress=plain
```

### 6.2 容器无法启动

**问题**: 容器启动后立即退出

**解决方案**:

```bash
# 查看容器日志
docker logs captcha-breaker

# 检查端口占用
netstat -tulpn | grep :8000

# 检查磁盘空间
df -h

# 检查内存
free -h
```

### 6.3 模型下载慢

**问题**: 首次启动时模型下载很慢

**解决方案**:

方式 1: 预下载模型到宿主机

```bash
# 创建模型目录
mkdir -p ~/.paddlex/official_models

# 从其他源下载模型 (如阿里云镜像)
# 然后挂载到容器
```

方式 2: 修改 docker-compose.yml 添加代理

```yaml
environment:
  - HTTP_PROXY=http://your-proxy:8080
  - HTTPS_PROXY=http://your-proxy:8080
```

### 6.4 内存不足

**问题**: OOM (Out of Memory) 错误

**解决方案**:

```bash
# 增加 swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 或者减少内存限制 (docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 2G
```

### 6.5 GPU 加速

**问题**: 如何使用 GPU 加速

**前提条件**:
- NVIDIA GPU
- 安装 NVIDIA 驱动
- 安装 NVIDIA Container Toolkit

**步骤**:

1. 安装 NVIDIA Container Toolkit:

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. 使用 GPU 版本的 docker-compose:

```bash
docker-compose -f docker-compose.gpu.yml up -d
```

---

## 7. 生产环境优化

### 7.1 多 Worker 模式

修改 `docker-compose.yml`:

```yaml
environment:
  - WORKERS=4
command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7.2 Nginx 反向代理

创建 `docker-compose.nginx.yml`:

```yaml
version: '3.8'

services:
  ocr-api:
    build: .
    expose:
      - "8000"
    networks:
      - ocr-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - ocr-api
    networks:
      - ocr-network

networks:
  ocr-network:
    driver: bridge
```

### 7.3 监控

使用 Prometheus + Grafana 监控：

```yaml
services:
  ocr-api:
    environment:
      - ENABLE_METRICS=true
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

### 7.4 日志管理

```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "5"
    compress: "true"
```

查看日志：

```bash
# 查看最近日志
docker-compose logs --tail=100

# 实时日志
docker-compose logs -f

# 导出日志
docker logs captcha-breaker > ocr-api.log 2>&1
```

---

## 8. 性能基准测试

### 测试环境
- **CPU**: Intel i7-10700 (8 核)
- **内存**: 16GB
- **系统**: Ubuntu 20.04
- **Docker**: 24.0

### 测试结果

| 指标 | 数值 |
|------|------|
| 首冷启动时间 | ~15-20 分钟 (模型下载) |
| 热启动时间 | ~30 秒 |
| 单次识别延迟 | ~3.5s (CPU) |
| 并发支持 | 5 QPS (单 worker) |
| 内存占用 | ~2GB (空闲) |
| 磁盘占用 | ~5GB (包含模型) |

### 优化建议

1. **预热模型**: 启动后发送一个测试请求，避免首次请求超时
2. **增加 Worker**: 根据 CPU 核心数调整 worker 数量
3. **使用 GPU**: 如果有 NVIDIA GPU，可提升 5-10x 性能

---

## 9. 故障排查

### 9.1 进入容器调试

```bash
docker exec -it captcha-breaker bash
```

### 9.2 检查服务状态

```bash
# 容器状态
docker ps -a | grep captcha

# 资源使用
docker stats captcha-breaker

# 端口监听
docker exec captcha-breaker netstat -tulpn
```

### 9.3 重新初始化

```bash
# 完全重建
docker-compose down -v --rmi all
docker system prune -af
docker-compose up -d --build
```

---

## 10. 更新服务

```bash
# 拉取最新代码
git pull

# 重新构建并启动
docker-compose up -d --build

# 验证版本
curl http://localhost:8000/
```
