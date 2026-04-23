#!/usr/bin/env python3
"""
测试当前版本准确率 - 简化版
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import smart_correct

# 之前测试中观察到的识别结果
test_results = [
    # (文件名, OCR识别结果, 期望结果)
    ("2zrw.png", "2zrw", "2zrw"),
    ("6QxJ.png", "QxJ", "6qxj"),
    ("78sq.png", "bs8L", "78sq"),
    ("aDkP.png", "aDkP", "adkp"),
    ("aYWW.png", "aYWW", "ayww"),
    ("aZWA.png", "zwa", "azwa"),
    ("BPAS.png", "BPAS", "bpas"),
    ("BRTf.png", "BRTf", "brtf"),
    ("BZvZ.png", "BZvZ", "bzvz"),
    ("cFY5.png", "cFY5", "cfy5"),
    ("cpVN.png", "cpVN", "cpvn"),
    ("G2tn.png", "G2tn", "g2tn"),
    ("Gtzf.png", "Gtzf", "gtzf"),
    ("H8bJ.png", "H8bJ", "h8bj"),
    ("HFvL.png", "HFvL", "hfvl"),
    ("hpLZ.png", "hpLZ", "hplz"),
    ("L2FK.png", "L2FK", "l2fk"),
    ("N4Qf.png", "N4Qf", "n4qf"),
    ("nDMN.png", "nDMN", "ndmn"),
    ("nfGH.png", "nfGH", "nfgh"),
    ("pWXA.png", "pWXA", "pwxa"),
    ("rGFE.png", "rGFE", "rgfe"),
    ("ssex.png", "ssex", "ssex"),
    ("tHn3.png", "tHn3", "thn3"),
    ("tmCm.png", "tmCm", "tmcm"),
    ("Ub35.png", "Ub35", "ub35"),
    ("udCS.png", "udCS", "udcs"),
    ("W7KW.png", "W7KW", "w7kw"),
    ("XZJu.png", "XZJu", "xzju"),
    ("Yvvr.png", "Yvvr", "yvvr"),
]

print("="*80)
print("CaptchaBreaker v7.0.0 准确率测试")
print("="*80 + "\n")

correct = 0
total = len(test_results)
failed = []

for filename, ocr_result, expected in test_results:
    corrected = smart_correct(ocr_result, expected_length=4)
    
    if corrected.lower() == expected or ocr_result.lower() == expected:
        correct += 1
        status = "✓"
    else:
        status = "✗"
        failed.append({
            'file': filename,
            'ocr': ocr_result,
            'corrected': corrected,
            'expected': expected
        })
    
    print(f"{status} {filename:12s} OCR: {ocr_result:10s} 纠错: {corrected:10s} 期望: {expected}")

print(f"\n{'='*80}")
print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
print(f"{'='*80}\n")

if failed:
    print(f"失败案例 ({len(failed)}):")
    for f in failed:
        print(f"  {f['file']:12s} OCR: {f['ocr']:10s} 纠错: {f['corrected']:10s} 期望: {f['expected']}")
    print()
