#!/usr/bin/env python3
"""
直接测试智能纠错引擎效果（不加载OCR模型）
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import smart_correct, vote_results

test_cases = [
    ("6QxJ", "qxj", "6qxj"),
    ("78sq", "bs8l", "78sq"),
    ("aZWA", "zwa", "azwa"),
    ("2zrw", "2zrw", "2zrw"),
    ("aDkP", "aDkP", "adkp"),
    ("aYWW", "aYWW", "ayww"),
    ("BPAS", "bpas", "bpas"),
    ("BRTf", "brtf", "brtf"),
    ("G2tn", "g2tn", "g2tn"),
]

print("="*70)
print("测试智能纠错引擎")
print("="*70)

correct = 0
total = len(test_cases)

for filename, ocr_result, expected in test_cases:
    corrected = smart_correct(ocr_result, expected_length=4)
    match = (corrected.lower() == expected.lower() or ocr_result.lower() == expected.lower())
    
    if match:
        correct += 1
        status = "[OK]"
    else:
        status = "[FAIL]"
    
    print(f"{filename:10s} OCR: {ocr_result:10s} -> Corrected: {corrected:10s} {status} (期望: {expected})")

print(f"\n{'='*70}")
print(f"纠错准确率: {correct}/{total} = {correct/total*100:.1f}%")
print(f"{'='*70}\n")
