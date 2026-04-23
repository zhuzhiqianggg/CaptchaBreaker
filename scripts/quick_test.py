#!/usr/bin/env python3
"""
快速测试当前版本准确率
"""

import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "samples"

from app.main import (
    preprocess_image_v1,
    preprocess_image_v2, 
    preprocess_image_v3,
    preprocess_image_v4,
    vote_results,
    smart_correct,
    parse_ocr_result
)
from paddleocr import PaddleOCR

print("Loading PaddleOCR model...")
ocr = PaddleOCR(lang='en')
print("Model loaded!\n")

imgs = sorted(DATA_DIR.glob("*.png"))

print(f"测试 {len(imgs)} 张图片...\n")
print("="*80)

total = 0
correct = 0
failed_list = []

for img_path in imgs:
    image = Image.open(img_path)
    expected = img_path.stem.lower().strip()
    total += 1
    
    strategies = [
        preprocess_image_v1,
        preprocess_image_v2,
        preprocess_image_v3,
        preprocess_image_v4,
    ]
    
    all_texts = []
    best_text = ""
    
    for strategy in strategies:
        processed, steps = strategy(image)
        temp_path = Path(__file__).parent.parent / "temp" / f"test_{img_path.stem}_{strategy.__name__}.png"
        temp_path.parent.mkdir(exist_ok=True)
        processed.save(temp_path)
        
        result = ocr.ocr(str(temp_path))
        texts, full_text = parse_ocr_result(result)
        all_texts.append(full_text)
        
        if not best_text:
            best_text = full_text
    
    voted = vote_results(all_texts)
    corrected = smart_correct(voted, expected_length=4)
    
    if corrected.lower() == expected or best_text.lower() == expected:
        correct += 1
        print(f"✓ {img_path.name:15s} 期望: {expected:8s} 识别: {voted:10s} 纠错: {corrected}")
    else:
        failed_list.append({
            'file': img_path.name,
            'expected': expected,
            'voted': voted,
            'corrected': corrected,
            'best': best_text
        })
        print(f"✗ {img_path.name:15s} 期望: {expected:8s} 识别: {voted:10s} 纠错: {corrected}")

print(f"\n{'='*80}")
print(f"测试结果: {correct}/{total}")
print(f"准确率: {correct/total*100:.1f}%")
print(f"{'='*80}\n")

if failed_list:
    print(f"失败案例 ({len(failed_list)}):")
    for f in failed_list:
        print(f"  {f['file']:15s} 期望: {f['expected']:8s} OCR: {f['voted']:10s} 纠错: {f['corrected']}")
    print()
