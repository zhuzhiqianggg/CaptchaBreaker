import requests
import os
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
IMAGE_DIR = "code_images"

def test_image_accuracy(image_path, language="en"):
    url = f"{BASE_URL}/ocr/upload"
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        data = {"language": language}
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        return {
            "status": "success",
            "recognized_text": result["full_text"],
            "confidence": max([t["confidence"] for t in result["texts"]]) if result["texts"] else 0,
            "text_count": len(result["texts"]),
            "time": elapsed,
            "details": result["texts"]
        }
    else:
        return {
            "status": "error",
            "error": response.text,
            "time": elapsed
        }

def main():
    print("=" * 80)
    print("OCR 验证码识别准确率测试")
    print("=" * 80)
    
    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    if not image_files:
        print(f"No PNG images found in {IMAGE_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to test\n")
    
    results = []
    total_success = 0
    total_error = 0
    total_confidence = 0
    total_time = 0
    
    for img_path in sorted(image_files):
        print(f"Testing: {img_path.name}...", end=" ", flush=True)
        
        result = test_image_accuracy(str(img_path))
        
        if result["status"] == "success":
            total_success += 1
            total_confidence += result["confidence"]
            total_time += result["time"]
            
            print(f"✓ '{result['recognized_text']}' (conf: {result['confidence']:.4f}, time: {result['time']:.2f}s)")
            
            results.append({
                "image": img_path.name,
                "recognized_text": result["recognized_text"],
                "confidence": result["confidence"],
                "text_count": result["text_count"],
                "time": result["time"],
                "status": "success"
            })
        else:
            total_error += 1
            total_time += result["time"]
            print(f"✗ Error: {result['error'][:50]}")
            
            results.append({
                "image": img_path.name,
                "status": "error",
                "error": result["error"],
                "time": result["time"]
            })
    
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    print(f"\n总图片数: {len(image_files)}")
    print(f"成功识别: {total_success}")
    print(f"识别失败: {total_error}")
    print(f"成功率: {total_success/len(image_files)*100:.1f}%")
    
    if total_success > 0:
        avg_confidence = total_confidence / total_success
        avg_time = total_time / len(image_files)
        print(f"\n平均置信度: {avg_confidence:.4f}")
        print(f"平均识别时间: {avg_time:.2f}s")
        print(f"总耗时: {total_time:.2f}s")
    
    print("\n详细结果:")
    print("-" * 80)
    for r in results:
        if r["status"] == "success":
            print(f"{r['image']:20s} | {r['recognized_text']:20s} | 置信度: {r['confidence']:.4f} | {r['text_count']}个文本 | {r['time']:.2f}s")
        else:
            print(f"{r['image']:20s} | 识别失败: {r['error'][:30]}")
    
    with open("ocr_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: ocr_test_results.json")

if __name__ == "__main__":
    main()
