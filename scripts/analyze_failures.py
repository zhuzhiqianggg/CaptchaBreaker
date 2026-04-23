#!/usr/bin/env python3
"""
深度分析失败模式
"""

import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "samples"

print("分析所有验证码的失败模式...")

from app.main import (
    preprocess_image_v1,
    preprocess_image_v2, 
    preprocess_image_v3,
    preprocess_image_v4,
    post_process_text,
    vote_results,
    parse_ocr_result
)
from paddleocr import PaddleOCR

print("Loading PaddleOCR model...")
ocr = PaddleOCR(lang='en')
print("Model loaded!\n")

imgs = sorted(DATA_DIR.glob("*.png"))

print("="*90)
print(f"{'图片':15s} {'期望':8s} {'最佳策略':12s} {'识别结果':12s} {'投票结果':12s} {'后处理':12s} {'错误类型'}")
print("="*90)

error_types = {
    '首字符丢失': 0,
    '尾字符丢失': 0,
    '中间字符丢失': 0,
    '字符混淆': 0,
    '额外字符': 0,
    '完全错误': 0,
}

total = 0
correct = 0

for img_path in imgs:
    image = Image.open(img_path)
    expected = img_path.stem.lower().strip()
    total += 1
    
    strategies = [
        ("v1", preprocess_image_v1),
        ("v2", preprocess_image_v2),
        ("v3", preprocess_image_v3),
        ("v4", preprocess_image_v4),
    ]
    
    all_texts = []
    best_text = ""
    best_strategy = ""
    best_confidence = -1
    
    for name, strategy in strategies:
        processed, steps = strategy(image)
        temp_path = Path(__file__).parent.parent / "temp" / f"analysis_{img_path.stem}_{name}.png"
        temp_path.parent.mkdir(exist_ok=True)
        processed.save(temp_path)
        
        result = ocr.ocr(str(temp_path))
        texts, full_text = parse_ocr_result(result)
        all_texts.append(full_text)
        
        avg_conf = 0.0
        if texts:
            avg_conf = sum(t.confidence for t in texts) / len(texts)
        
        if avg_conf > best_confidence:
            best_confidence = avg_conf
            best_text = full_text
            best_strategy = name
    
    voted = vote_results(all_texts)
    final = post_process_text(voted)
    
    if final == expected:
        correct += 1
        error_type = "✓"
    elif best_text.lower() == expected:
        correct += 1
        error_type = "✓(best)"
    else:
        # 分析错误类型
        if len(final) < len(expected):
            if not expected.startswith(final[:2]):
                error_types['首字符丢失'] += 1
                error_type = "首字符丢失"
            elif not expected.endswith(final[-2:] if len(final) >= 2 else final):
                error_types['尾字符丢失'] += 1
                error_type = "尾字符丢失"
            else:
                error_types['中间字符丢失'] += 1
                error_type = "中间字符丢失"
        elif len(final) > len(expected):
            error_types['额外字符'] += 1
            error_type = "额外字符"
        else:
            # 长度相同但内容不同
            diff_count = sum(1 for a, b in zip(final, expected) if a != b)
            if diff_count <= 2:
                error_types['字符混淆'] += 1
                error_type = f"字符混淆({diff_count})"
            else:
                error_types['完全错误'] += 1
                error_type = "完全错误"
    
    print(f"{img_path.name:15s} {expected:8s} {best_strategy:12s} {best_text:12s} {voted:12s} {final:12s} {error_type}")

print(f"\n{'='*90}")
print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
print(f"\n错误类型统计:")
for error_type, count in error_types.items():
    if count > 0:
        print(f"  {error_type}: {count} ({count/total*100:.1f}%)")
print(f"{'='*90}\n")
