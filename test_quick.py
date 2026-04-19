import requests
import os
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
IMAGE_DIR = "code_images"

def normalize_text(text):
    return text.strip().replace(" ", "").replace("-", "").lower()

def test_image(image_path):
    ground_truth = os.path.splitext(image_path.name)[0].lower()
    url = f"{BASE_URL}/ocr/upload"
    
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/png")}
        data = {"language": "en"}
        start = time.time()
        try:
            response = requests.post(url, files=files, data=data, timeout=30)
            elapsed = time.time() - start
        except Exception as e:
            return {"image": image_path.name, "expected": ground_truth, "got": "ERROR", "match": False, "time": elapsed}
    
    if response.status_code == 200:
        result = response.json()
        recognized = normalize_text(result["full_text"])
        match = (recognized == ground_truth)
        return {
            "image": image_path.name,
            "expected": ground_truth,
            "got": recognized,
            "confidence": result["texts"][0]["confidence"] if result["texts"] else 0,
            "match": match,
            "time": elapsed
        }
    else:
        return {"image": image_path.name, "expected": ground_truth, "got": "ERROR", "match": False, "time": elapsed}

def main():
    print("OCR 验证码真实准确率快速测试")
    print("=" * 60)
    
    images = sorted(Path(IMAGE_DIR).glob("*.png"))[:10]
    print(f"测试前 {len(images)} 张图片...\n")
    
    results = []
    correct = 0
    
    for img in images:
        r = test_image(img)
        results.append(r)
        status = "✓" if r["match"] else "✗"
        print(f"{status} {r['image']:15s} | 期望: {r['expected']:8s} | 识别: {r['got']:8s} | {r['time']:.1f}s")
        if r["match"]:
            correct += 1
    
    accuracy = correct / len(results) * 100 if results else 0
    print(f"\n{'='*60}")
    print(f"测试结果 (前{len(results)}张):")
    print(f"  正确: {correct}/{len(results)}")
    print(f"  准确率: {accuracy:.1f}%")
    
    with open("ocr_quick_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: ocr_quick_results.json")

if __name__ == "__main__":
    main()
