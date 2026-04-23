#!/usr/bin/env python3
"""
PaddleOCR Fine-tuning 脚本
使用增强后的验证码数据训练识别模型
"""

import os
import shutil
from pathlib import Path
import paddle
from paddleocr import PaddleOCR

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "finetune"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = PROJECT_DIR / "output" / "finetune"

def create_dict_file():
    """创建字符字典文件"""
    label_file = DATA_DIR / "rec_gt_train.txt"
    char_set = set()
    
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            _, label = line.strip().split("\t")
            for char in label:
                char_set.add(char)
    
    char_dict = sorted(list(char_set))
    
    dict_file = DATA_DIR / "char_dict.txt"
    with open(dict_file, "w", encoding="utf-8") as f:
        for char in char_dict:
            f.write(f"{char}\n")
    
    print(f"✓ 字符字典已生成: {dict_file}")
    print(f"  字符集 ({len(char_dict)} 个): {''.join(char_dict)}")
    
    return dict_file


def split_train_val(ratio=0.8):
    """划分训练集和验证集"""
    label_file = DATA_DIR / "rec_gt_train.txt"
    
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    import random
    random.shuffle(lines)
    
    split_idx = int(len(lines) * ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]
    
    train_file = DATA_DIR / "rec_gt_train_split.txt"
    val_file = DATA_DIR / "rec_gt_val.txt"
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_lines)
    
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_lines)
    
    print(f"✓ 数据集已划分:")
    print(f"  训练集: {len(train_lines)} 条 -> {train_file}")
    print(f"  验证集: {len(val_lines)} 条 -> {val_file}")
    
    return train_file, val_file


def main():
    print("=" * 70)
    print("PaddleOCR Fine-tuning 准备")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/3] 生成字符字典...")
    dict_file = create_dict_file()
    
    print("\n[2/3] 划分训练集和验证集...")
    train_file, val_file = split_train_val(ratio=0.8)
    
    print("\n[3/3] 创建训练配置...")
    
    print(f"\n{'='*70}")
    print("数据准备完成！")
    print(f"{'='*70}")
    print(f"\n训练数据目录: {TRAIN_DIR}")
    print(f"训练标注文件: {train_file}")
    print(f"验证标注文件: {val_file}")
    print(f"字符字典: {dict_file}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"\n下一步: 运行训练脚本")
    print(f"  python scripts/train_model.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
