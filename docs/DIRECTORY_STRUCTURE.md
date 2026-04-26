# CaptchaBreaker 目录结构说明

## 📁 完整目录结构

```
CaptchaBreaker/
│
├── app/                          # 🚀 主应用程序
│   ├── __init__.py               # Python 包初始化
│   └── main.py                   # FastAPI 服务 + OCR 核心逻辑
│
├── config/                       # ⚙️ 配置文件
│   └── auto_train.yaml           # 自动训练配置
│
├── data/                         # 📊 数据目录（核心）
│   ├── samples/                  # 测试用验证码图片（30张，已知答案）
│   │   ├── 2zrw.png              #   例如：2zrw.png 答案就是 "2zrw"
│   │   ├── 6QxJ.png              #   用于测试和评估模型准确率
│   │   └── ...
│   │
│   ├── collected/                # 📥 自动收集的用户数据（训练数据来源）
│   │   ├── original/             #   原始验证码图片
│   │   ├── v1_light/             #   轻量预处理后的图片
│   │   ├── v2_medium/            #   中等增强后的图片
│   │   ├── v3_binary/            #   二值化处理后的图片
│   │   ├── v4_denoise/           #   降噪处理后的图片
│   │   └── metadata/             #   元数据（JSON，包含识别结果）
│   │                             #   ⚠️ 这是最重要的训练数据来源！
│   │
│   └── finetune/                 # 🎯 训练准备区（用于 fine-tuning）
│       ├── train/                #   训练图片（数据增强后的）
│       ├── char_dict.txt         #   字符字典
│       ├── rec_gt_train.txt      #   训练标注文件（图片名\t文本）
│       └── rec_gt_val.txt        #   验证标注文件
│
├── output/                       # 📤 输出目录（训练结果）
│   └── finetune/                 #   训练好的模型保存在这里
│       ├── best_accuracy.pdparams  # 最佳模型
│       ├── best_accuracy.pdopt     # 优化器状态
│       └── ...
│
├── models/                       # 🤖 模型目录（推理用）
│   └── custom/                   #   自定义训练好的模型（部署用）
│
├── docs/                         # 📖 文档
│   ├── ARCHITECTURE.md           # 架构文档
│   ├── OPTIMIZATION.md           # 优化指南
│   ├── API.md                    # API 文档
│   ├── DOCKER.md                 # Docker 部署文档
│   ├── MODEL_TRAINING_GUIDE.md   # 大模型训练指南
│   ├── CAPTCHA_TRAINING_GUIDE.md # 验证码训练指南
│   └── AUTO_TRAINING_GUIDE.md    # 自动训练指南
│
├── scripts/                      # 🛠️ 脚本工具
│   ├── run_tests.py              # 统一测试入口（推荐使用）
│   ├── test_accuracy.py          # 准确率测试
│   ├── test_single.py            # 单图测试
│   ├── test_correction.py        # 纠错引擎测试
│   ├── analyze_failures.py       # 失败分析
│   ├── auto_train.py             # 手动触发自动训练
│   ├── auto_train_daemon.py      # 训练守护进程
│   ├── prepare_training.py       # 准备训练数据
│   ├── train_model.py            # 执行模型训练
│   ├── augment_data.py           # 数据增强
│   └── generate_labels.py        # 标签生成
│
├── temp/                         # 🗑️ 临时文件（可安全删除）
├── logs/                         # 📝 日志文件
│   └── auto_train.log            # 自动训练日志
│
├── Dockerfile                    # 🐳 Docker 镜像配置
├── Dockerfile.gpu               # GPU 版 Docker
├── docker-compose.yml           # Docker 编排
├── docker-compose.gpu.yml       # GPU 版编排
├── .gitignore                   # Git 忽略规则
├── .dockerignore               # Docker 忽略规则
├── run.py                       # 启动入口
├── requirements.txt             # Python 依赖
└── README.md                    # 项目文档
```

---

## 📋 核心目录说明

### 1. data/ - 数据目录（训练核心）

**职责**：存储所有训练相关的数据

| 子目录 | 作用 | 数据来源 | 是否 Git 跟踪 |
|--------|------|---------|--------------|
| `samples/` | 测试验证码 | 手动准备 | ✅ 是 |
| `collected/` | 用户真实数据 | **自动收集** | ❌ 否 |
| `finetune/` | 训练准备区 | 从 collected/ 转换 | ❌ 否 |

#### 数据流向

```
用户使用 OCR
    ↓
自动保存到 data/collected/
    ↓
达到训练阈值
    ↓
prepare_training.py 转换格式
    ↓
保存到 data/finetune/
    ↓
train_model.py 训练
    ↓
模型保存到 output/finetune/
```

#### 数据收集机制

每次用户请求 OCR 识别，系统会自动保存：

1. **原始图片** → `data/collected/original/{image_id}.png`
2. **4 种预处理图片** → `data/collected/v1_light/`, `v2_medium/`, `v3_binary/`, `v4_denoise/`
3. **元数据** → `data/collected/metadata/{image_id}.json`

元数据包含：
- 识别结果
- 使用的预处理策略
- 各策略的识别结果
- 纠错后的文本

### 2. output/ - 输出目录（训练结果）

**职责**：存储训练过程中产生的输出

| 子目录 | 内容 |
|--------|------|
| `finetune/` | 训练好的模型文件 |
| `logs/` | 训练日志 |

### 3. models/ - 模型目录（推理用）

**职责**：存储用于推理的模型

| 子目录 | 内容 |
|--------|------|
| `custom/` | 自定义训练好的模型（部署用） |
| `custom_backup/` | 旧模型备份 |

---

## 🔄 数据流完整示例

### 用户请求流程

```
1. 用户发送验证码图片
   ↓
2. FastAPI 接收请求
   ↓
3. OCR 识别（4 种预处理策略）
   ↓
4. 投票 + 纠错 → 返回结果
   ↓
5. 自动保存数据到 data/collected/
   ✓ 原始图片
   ✓ 预处理图片
   ✓ 识别结果（JSON）
```

### 自动训练流程

```
1. 监控 data/collected/ 中的数据量
   ↓
2. 达到阈值（如 100 张）
   ↓
3. prepare_training.py 执行：
   ✓ 复制图片到 data/finetune/train/
   ✓ 生成标注文件 rec_gt_train.txt
   ✓ 生成字符字典 char_dict.txt
   ↓
4. train_model.py 执行：
   ✓ 加载预训练模型
   ✓ 在 data/finetune/ 上训练
   ✓ 保存最佳模型到 output/finetune/
   ↓
5. 评估模型准确率
   ↓
6. 如果准确率达标：
   ✓ 复制模型到 models/custom/
   ✓ 热重载模型（无需重启）
```

---

## 📊 数据量统计

### 查看已收集的数据

```bash
# Linux
echo "已收集样本数:"
find data/collected/original -name "*.png" | wc -l

# Windows PowerShell
Write-Host "已收集样本数:"
(Get-ChildItem -Path "data\collected\original" -Filter "*.png").Count
```

### 训练数据统计

```bash
# 查看训练准备区
ls -l data/finetune/train/ | wc -l

# 查看标注文件
wc -l data/finetune/rec_gt_train.txt
```

---

## 💡 最佳实践

1. **不要手动修改 `data/collected/`**
   - 这是自动收集的，手动修改可能导致元数据不一致

2. **定期清理旧数据**
   ```bash
   # 清理超过 90 天的数据
   find data/collected -type f -mtime +90 -delete
   ```

3. **备份重要数据**
   - `data/collected/` 是最重要的数据源
   - 定期备份到云存储或外部硬盘

4. **训练前检查数据质量**
   ```bash
   # 查看收集的样本
   ls -lt data/collected/original | head -20
   
   # 查看元数据
   cat data/collected/metadata/xxx.json
   ```

---

## 🔧 常用命令

### 数据管理

```bash
# 查看已收集样本数
find data/collected/original -name "*.png" | wc -l

# 查看最新的样本
ls -lt data/collected/original | head -20

# 清理旧数据（超过 30 天）
find data/collected -type f -mtime +30 -delete
```

### 训练相关

```bash
# 准备训练数据
python scripts/prepare_training.py

# 执行训练
python scripts/train_model.py

# 手动触发自动训练
python scripts/auto_train.py
```

### 测试相关

```bash
# 运行测试
python scripts/run_tests.py api

# 单图测试
python scripts/run_tests.py single data/samples/2zrw.png
```

---

## ❓ 常见问题

### Q: `data/collected/` 和 `data/finetune/` 有什么区别？

**A**: 
- `collected/` 是**原始数据**，系统自动收集
- `finetune/` 是**训练准备区**，从 collected/ 转换而来

### Q: 可以删除 `data/collected/` 吗？

**A**: **不建议**！这是最重要的训练数据来源。如果删除，需要重新收集数据才能训练。

### Q: `output/` 目录可以删除吗？

**A**: **可以**。`output/` 只包含训练输出，删除后重新训练即可。但会丢失训练好的模型。

### Q: 如何查看已收集了多少数据？

**A**: 
```bash
# Linux
find data/collected/original -name "*.png" | wc -l

# Windows
(Get-ChildItem -Path "data\collected\original" -Filter "*.png").Count
```
