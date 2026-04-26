# CaptchaBreaker 自动训练使用指南

## 一、快速开始

### 1.1 自动训练流程

```
用户使用 → 数据自动收集 → 达到阈值 → 自动训练 → 自动评估 → 自动部署
```

**整个过程无需人工干预！**

### 1.2 三种使用方式

#### 方式 1: 手动训练（适合测试）

```bash
# 检查当前数据量并准备训练
python scripts/auto_train.py

# 自定义阈值（例如 50 张）
python scripts/auto_train.py --threshold 50
```

#### 方式 2: 自动训练守护进程（推荐）

```bash
# 启动自动训练守护进程
python scripts/auto_train_daemon.py

# 自定义参数
python scripts/auto_train_daemon.py \
    --threshold 200 \
    --interval 1800 \
    --epochs 50
```

#### 方式 3: 定时任务（生产环境）

```bash
# Linux: 每天凌晨 2 点训练
crontab -e
# 添加: 0 2 * * * cd /path/to/CaptchaBreaker && python scripts/auto_train.py

# Windows: 使用任务计划程序
# 见下方详细说明
```

---

## 二、数据收集

### 2.1 自动收集机制

系统会自动收集每次识别的数据，**无需任何配置**：

```
用户请求 → OCR 识别 → 保存到 data/collected/
```

**保存的内容包括：**
- 原始验证码图片
- 4 种预处理后的图片（轻量、中等、二值、降噪）
- 识别结果和元数据（JSON 格式）

### 2.2 查看收集的数据

```bash
# 查看已收集的样本数量
find data/collected/original -name "*.png" | wc -l

# 查看最近的样本
ls -lt data/collected/original | head -20

# 查看元数据
cat data/collected/metadata/abc123.json
```

Windows PowerShell:
```powershell
# 查看已收集的样本数量
(Get-ChildItem -Path "data\collected\original" -Filter "*.png").Count

# 查看最近的样本
Get-ChildItem -Path "data\collected\original" | Sort-Object LastWriteTime -Descending | Select-Object -First 20
```

### 2.3 数据清理

```bash
# 清理超过 30 天的旧数据
find data/collected -type f -mtime +30 -delete

# 或者限制最大样本数
python scripts/auto_train.py --max_samples 10000
```

---

## 三、训练配置

### 3.1 配置文件

编辑 `config/auto_train.yaml` 配置自动训练参数：

```yaml
data_collection:
  min_samples: 100          # 达到 100 张时触发训练

training:
  epochs: 100               # 训练 100 轮
  batch_size: 32            # 批次大小 32
  learning_rate: 0.0001     # 学习率
```

### 3.2 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| **min_samples** | 触发训练的最小样本数 | 100-500 |
| **epochs** | 训练轮数 | 50-200 |
| **batch_size** | 批次大小 | 16-64（根据显存） |
| **learning_rate** | 学习率 | 1e-4 ~ 5e-4 |
| **min_accuracy** | 最低准确率阈值 | 0.85-0.95 |
| **auto_deploy** | 是否自动部署 | true/false |

---

## 四、训练执行

### 4.1 准备训练数据

```bash
python scripts/prepare_training.py
```

**生成文件：**
- `data/finetune/rec_gt_train.txt` - 训练标注文件
- `data/finetune/char_dict.txt` - 字符字典
- `data/finetune/train/` - 训练图片目录

### 4.2 执行训练

```bash
python scripts/train_model.py
```

训练过程输出示例：
```
Epoch [10/100], Loss: 0.1234, Acc: 0.85
Epoch [20/100], Loss: 0.0891, Acc: 0.89
Epoch [30/100], Loss: 0.0654, Acc: 0.92
...
```

### 4.3 使用 PaddleX 训练（推荐）

```bash
# 安装 PaddleX
pip install paddlex

# 一键训练
paddlex --task OCR --train \
    --dataset ./data/finetune \
    --save_dir ./output/finetune \
    --epochs 100
```

---

## 五、模型部署

### 5.1 手动部署

```bash
# 1. 备份旧模型
cp -r models/custom models/custom_backup

# 2. 部署新模型
cp output/finetune/best_accuracy.pdparams models/custom/

# 3. 重启服务
docker-compose restart
```

### 5.2 自动部署

配置文件中设置 `auto_deploy: true`，训练完成后会自动部署：

```yaml
deployment:
  auto_deploy: true          # 自动部署
  backup_old: true           # 备份旧模型
  rollback_on_fail: true     # 失败时回滚
```

### 5.3 热重载（无需重启）

```bash
# 调用 API 重新加载模型
curl -X POST http://localhost:8000/admin/reload_model
```

---

## 六、监控和日志

### 6.1 查看训练日志

```bash
# 实时查看日志
tail -f logs/auto_train.log

# 查看最近的日志
tail -100 logs/auto_train.log
```

Windows:
```powershell
Get-Content logs/auto_train.log -Wait -Tail 100
```

### 6.2 日志格式

```
2024-01-01 02:00:00 [INFO] ============================================================
2024-01-01 02:00:00 [INFO] 开始训练周期 #1
2024-01-01 02:00:00 [INFO] 已收集样本: 150
2024-01-01 02:00:01 [INFO] 准备训练数据...
2024-01-01 02:00:02 [INFO] ✓ 已准备 150 张训练图片
2024-01-01 02:00:03 [INFO] 开始模型训练...
2024-01-01 02:05:00 [INFO] ✓ 训练完成
2024-01-01 02:05:01 [INFO] ✓ 模型准确率: 92.5%
2024-01-01 02:05:02 [INFO] 开始部署模型...
2024-01-01 02:05:03 [INFO] ✓ 模型已部署到: models/custom
2024-01-01 02:05:04 [INFO] ✓ 训练周期完成! 累计训练 1 次
```

### 6.3 监控训练效果

```bash
# 查看训练统计
python scripts/train_stats.py
```

输出示例：
```
====================================
训练统计
====================================
累计训练次数:    5
最新准确率:    94.2%
最佳准确率:    94.2%
平均准确率:    91.8%
总训练时间:    2.5 小时
====================================
```

---

## 七、Windows 定时任务设置

### 7.1 使用 PowerShell 创建定时任务

```powershell
# 创建定时任务
$action = New-ScheduledTaskAction `
    -Execute "python" `
    -Argument "scripts\auto_train.py" `
    -WorkingDirectory "C:\dev_project\ocr"

$trigger = New-ScheduledTaskTrigger -Daily -At 2:00AM

Register-ScheduledTask `
    -TaskName "CaptchaBreaker Auto Training" `
    -Action $action `
    -Trigger $trigger `
    -User "SYSTEM" `
    -RunLevel Highest
```

### 7.2 查看定时任务

```powershell
# 查看任务状态
Get-ScheduledTask -TaskName "CaptchaBreaker Auto Training"

# 禁用任务
Disable-ScheduledTask -TaskName "CaptchaBreaker Auto Training"

# 启用任务
Enable-ScheduledTask -TaskName "CaptchaBreaker Auto Training"

# 删除任务
Unregister-ScheduledTask -TaskName "CaptchaBreaker Auto Training" -Confirm:$false
```

---

## 八、Linux 定时任务设置

### 8.1 使用 Cron

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天凌晨 2 点训练）
0 2 * * * cd /path/to/CaptchaBreaker && python scripts/auto_train.py >> logs/auto_train.log 2>&1

# 保存退出
```

### 8.2 查看 Cron 日志

```bash
# 查看 cron 日志
grep CRON /var/log/syslog

# 查看训练日志
tail -f /path/to/CaptchaBreaker/logs/auto_train.log
```

---

## 九、故障排除

### 9.1 训练不触发

**问题**：数据量已达阈值，但没有触发训练

**排查步骤：**
```bash
# 1. 检查数据量
find data/collected/original -name "*.png" | wc -l

# 2. 检查日志
tail -100 logs/auto_train.log

# 3. 手动触发
python scripts/auto_train.py
```

### 9.2 训练失败

**问题**：训练过程中报错

**排查步骤：**
```bash
# 1. 查看详细日志
tail -500 logs/auto_train.log

# 2. 检查训练数据
cat data/finetune/rec_gt_train.txt

# 3. 检查显存
nvidia-smi

# 4. 减小 batch_size
python scripts/auto_train.py --batch_size 16
```

### 9.3 部署失败

**问题**：训练完成但部署失败

**排查步骤：**
```bash
# 1. 检查模型文件
ls -lh output/finetune/

# 2. 检查部署目录
ls -lh models/custom/

# 3. 回滚到旧模型
rm -rf models/custom
cp -r models/custom_backup models/custom

# 4. 重启服务
docker-compose restart
```

---

## 十、最佳实践

1. **从小阈值开始**：先用 100 张测试训练流程
2. **监控训练日志**：定期检查日志确保训练正常
3. **保留备份**：始终开启 `backup_old: true`
4. **逐步增加阈值**：根据数据增长速度调整
5. **定期人工审核**：每周检查一次训练效果
6. **设置告警**：准确率低于阈值时发送通知
7. **A/B 测试**：新旧模型并行测试，确认效果提升后再切换

---

## 十一、常见问题 FAQ

### Q1: 需要多少数据才能开始训练？

**A**: 建议至少 100 张，但 500+ 张效果更好。

### Q2: 训练需要多长时间？

**A**: 取决于数据量和硬件：
- 100 张图片，CPU: ~30 分钟
- 100 张图片，GPU: ~5 分钟
- 1000 张图片，GPU: ~30 分钟

### Q3: 训练会影响正在运行的服务吗？

**A**: 不会。训练在后台执行，不影响现有服务。部署时可以选择热重载或重启。

### Q4: 如何知道训练是否有效？

**A**: 查看训练日志中的准确率指标，或使用测试脚本验证：
```bash
python scripts/run_tests.py api
```

### Q5: 训练后准确率反而下降怎么办？

**A**: 
1. 系统会自动回滚到旧模型（如果开启了 `rollback_on_fail`）
2. 降低学习率，减少训练轮数
3. 检查数据标注是否正确

---

## 相关文档

- [大模型训练通用指南](MODEL_TRAINING_GUIDE.md)
- [验证码识别训练指南](CAPTCHA_TRAINING_GUIDE.md)
- [Docker 部署文档](DOCKER.md)
- [架构文档](ARCHITECTURE.md)
