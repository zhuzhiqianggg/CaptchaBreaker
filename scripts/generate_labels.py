#!/usr/bin/env python3
"""
生成 PaddleOCR 训练标注文件
"""

import os
from pathlib import Path

TRAIN_DIR = Path(__file__).parent.parent / "data" / "finetune" / "train"
OUTPUT_FILE = TRAIN_DIR.parent / "rec_gt_train.txt"

def main():
    print("生成训练标注文件...")
    
    images = sorted(TRAIN_DIR.glob("*.png"))
    if not images:
        print("未找到训练图片！")
        return
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for img_path in images:
            filename = img_path.name
            label = filename.split("_")[0]
            f.write(f"{filename}\t{label}\n")
    
    print(f"✓ 标注文件已生成: {OUTPUT_FILE}")
    print(f"  总计: {len(images)} 条标注")
    print(f"\n前10条标注:")
    
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(f"  {line.strip()}")

if __name__ == "__main__":
    main()
