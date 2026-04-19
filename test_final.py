import requests
import os
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
IMAGE_DIR = "code_images"

print("=" * 70)
print("OCR 验证码真实准确率测试")
print("基准: 图片文件名 = 验证码内容")
print("=" * 70)

images = sorted(Path(IMAGE_DIR).glob("*.png"))
print(f"\n共 {len(images)} 张图片\n")

results = []
correct = 0
partial = 0

for img_path in images:
    expected = os.path.splitext(img_path.name)[0].lower()
    print(f"测试: {img_path.name} (期望: {expected})", end="... ", flush=True)
    
    start = time.time()
    try:
        with open(img_path, "rb") as f:
            response = requests.post(
                f"{BASE_URL}/ocr/upload",
                files={"file": (img_path.name, f, "image/png")},
                data={"language": "en"},
                timeout=30
            )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            recognized = data["full_text"].strip().replace(" ", "").lower()
            confidence = max([t["confidence"] for t in data["texts"]]) if data["texts"] else 0
            
            if recognized == expected:
                print(f"✓ '{recognized}' (conf: {confidence:.3f})")
                correct += 1
            elif any(c in recognized for c in expected) and len(recognized) > 0:
                print(f"⚠ '{recognized}' (conf: {confidence:.3f})")
                partial += 1
            else:
                print(f"✗ '{recognized}' (conf: {confidence:.3f})")
            
            results.append({
                "image": img_path.name,
                "expected": expected,
                "recognized": recognized,
                "confidence": confidence,
                "match": recognized == expected,
                "partial": recognized != expected and len(recognized) > 0
            })
        else:
            print(f"✗ Error: {response.status_code}")
            results.append({"image": img_path.name, "expected": expected, "recognized": "ERROR", "match": False})
    except Exception as e:
        print(f"✗ Exception: {str(e)[:30]}")
        results.append({"image": img_path.name, "expected": expected, "recognized": "ERROR", "match": False})

print(f"\n{'='*70}")
print(f"测试结果")
print(f"{'='*70}")
print(f"总图片数: {len(images)}")
print(f"完全正确: {correct} ({correct/len(images)*100:.1f}%)")
print(f"部分匹配: {partial} ({partial/len(images)*100:.1f}%)")
print(f"完全错误: {len(images) - correct - partial} ({(len(images) - correct - partial)/len(images)*100:.1f}%)")
print(f"\n真实准确率: {correct/len(images)*100:.1f}%")

print(f"\n详细结果:")
print(f"{'图片':15s} | {'期望':8s} | {'识别':10s} | {'状态':8s}")
print("-" * 55)
for r in results:
    status = "✓ 正确" if r["match"] else ("⚠ 部分" if r.get("partial") else "✗ 错误")
    print(f"{r['image']:15s} | {r['expected']:8s} | {r.get('recognized','?'):10s} | {status}")

with open("ocr_test_results_final.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n结果已保存到: ocr_test_results_final.json")
