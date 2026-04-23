#!/usr/bin/env python3
"""
测试 v7.0.0 智能纠错引擎
"""

import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "samples"

print("="*70)
print("CaptchaBreaker v7.0.0 智能纠错引擎测试")
print("="*70)

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

print("\nLoading PaddleOCR model...")
ocr = PaddleOCR(lang='en')
print("Model loaded!\n")

imgs = sorted(DATA_DIR.glob("*.png"))

print(f"测试 {len(imgs)} 张图片...\n")
print("="*90)
print(f"{'图片':15s} {'期望':8s} {'识别':12s} {'纠错':12s} {'结果'}")
print("="*90)

total = 0
correct = 0
corrected_count = 0
results = []

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
    best_confidence = -1
    
    for strategy in strategies:
        processed, steps = strategy(image)
        temp_path = Path(__file__).parent.parent / "temp" / f"test_{img_path.stem}_{strategy.__name__}.png"
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
    
    voted = vote_results(all_texts)
    corrected = smart_correct(voted, expected_length=4)
    
    if corrected == expected:
        correct += 1
        status = "[OK]"
    elif best_text.lower() == expected:
        correct += 1
        status = "[OK]"
    else:
        status = "[FAIL]"
    
    if corrected != voted and voted.lower() != expected and corrected.lower() == expected:
        corrected_count += 1
        status = "[OK*]"
    
    print(f"{img_path.name:15s} {expected:8s} {voted:12s} {corrected:12s} {status}")
    results.append({
        'file': img_path.name,
        'expected': expected,
        'voted': voted,
        'corrected': corrected,
        'match': (corrected == expected or best_text.lower() == expected)
    })

print(f"\n{'='*90}")
print(f"测试结果: {correct}/{total} 正确")
print(f"准确率: {correct/total*100:.1f}%")
if corrected_count > 0:
    print(f"智能纠错成功: {corrected_count} 个")
print(f"{'='*90}\n")

if correct < total:
    print("失败案例:")
    for r in results:
        if not r['match']:
            print(f"  {r['file']:15s} 期望: {r['expected']:8s} 识别: {r['voted']:12s} 纠错: {r['corrected']}")
    print()
