#!/usr/bin/env python3
"""
自动训练脚本
当收集的训练数据达到阈值时，自动触发 PaddleOCR 模型 fine-tuning
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(__file__).parent.parent
COLLECTED_DIR = PROJECT_DIR / "data" / "collected"
FINETUNE_DIR = PROJECT_DIR / "data" / "finetune"
OUTPUT_DIR = PROJECT_DIR / "output" / "finetune"

TRAIN_THRESHOLD = 100

def check_collected_data():
    """检查已收集的数据量"""
    original_dir = COLLECTED_DIR / "original"
    if not original_dir.exists():
        return 0, []
    
    images = list(original_dir.glob("*.png"))
    return len(images), images

def prepare_training_data(collected_images):
    """准备训练数据格式"""
    print("正在准备训练数据...")
    
    train_dir = FINETUNE_DIR / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    
    label_lines = []
    copied_count = 0
    
    for img_path in collected_images:
        metadata_path = COLLECTED_DIR / "metadata" / f"{img_path.stem}.json"
        if not metadata_path.exists():
            continue
        
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            final_text = metadata.get("final_text", "")
            if not final_text or len(final_text) < 3:
                continue
            
            dest_path = train_dir / img_path.name
            shutil.copy2(img_path, dest_path)
            
            label_lines.append(f"{img_path.name}\t{final_text}\n")
            copied_count += 1
            
        except Exception as e:
            print(f"跳过 {img_path.name}: {e}")
    
    label_file = FINETUNE_DIR / "rec_gt_train.txt"
    with open(label_file, "w", encoding="utf-8") as f:
        f.writelines(label_lines)
    
    print(f"✓ 已复制 {copied_count} 张图片到训练目录")
    print(f"✓ 标注文件已生成: {label_file}")
    
    return copied_count > 0

def create_char_dict():
    """创建字符字典"""
    label_file = FINETUNE_DIR / "rec_gt_train.txt"
    if not label_file.exists():
        return False
    
    char_set = set()
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                for char in parts[1]:
                    char_set.add(char)
    
    dict_file = FINETUNE_DIR / "char_dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for char in sorted(char_set):
            f.write(f"{char}\n")
    
    print(f"✓ 字符字典已生成: {len(char_set)} 个字符")
    return True

def train_model():
    """执行模型训练"""
    print("\n" + "="*70)
    print("开始模型训练")
    print("="*70)
    
    print("\n建议的训练命令:")
    print("\n方式 1: 使用 PaddleX (推荐)")
    print("  paddlex --task OCR --train --dataset ./data/finetune")
    
    print("\n方式 2: 使用 PaddleOCR 原生训练")
    print("  python tools/train.py \\")
    print("    -c configs/rec/rec_mv3_en_PP-OCRv5_train.yml \\")
    print("    -o Global.pretrained_model=./pretrain_models/en_PP-OCRv5_mobile_rec/best_accuracy")
    
    print("\n" + "="*70)
    print("训练配置:")
    print(f"  训练数据: {FINETUNE_DIR / 'train'}")
    print(f"  标注文件: {FINETUNE_DIR / 'rec_gt_train.txt'}")
    print(f"  字符字典: {FINETUNE_DIR / 'char_dict.txt'}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("="*70 + "\n")

def deploy_trained_model():
    """部署训练好的模型"""
    best_model = OUTPUT_DIR / "best_accuracy.pdparams"
    if not best_model.exists():
        print("❌ 未找到训练好的模型")
        return False
    
    print("✓ 训练完成，部署模型...")
    
    deploy_dir = PROJECT_DIR / "models" / "custom"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    
    for ext in [".pdparams", ".pdopt", ".pdstates"]:
        src = OUTPUT_DIR / f"best_accuracy{ext}"
        if src.exists():
            shutil.copy2(src, deploy_dir / f"rec_model{ext}")
    
    print(f"✓ 模型已部署到: {deploy_dir}")
    return True

def auto_train():
    """自动训练主流程"""
    print("="*70)
    print("CaptchaBreaker 自动训练")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    count, images = check_collected_data()
    print(f"\n已收集数据: {count} 张")
    print(f"训练阈值: {TRAIN_THRESHOLD} 张")
    
    if count < TRAIN_THRESHOLD:
        print(f"\n⏳ 数据不足，还需收集 {TRAIN_THRESHOLD - count} 张")
        return False
    
    print(f"\n✅ 达到训练阈值，开始准备训练数据...")
    
    if not prepare_training_data(images):
        print("❌ 准备训练数据失败")
        return False
    
    if not create_char_dict():
        print("❌ 创建字符字典失败")
        return False
    
    train_model()
    
    print("\n" + "="*70)
    print("训练准备完成！")
    print("请手动运行训练命令，或在配置中设置自动训练")
    print("="*70 + "\n")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="自动训练脚本")
    parser.add_argument("--threshold", type=int, default=TRAIN_THRESHOLD, help="训练数据阈值")
    parser.add_argument("--auto-deploy", action="store_true", help="训练完成后自动部署")
    
    args = parser.parse_args()
    TRAIN_THRESHOLD = args.threshold
    
    auto_train()
