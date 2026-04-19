import requests
import os
import json
import time
import re
from pathlib import Path

BASE_URL = "http://localhost:8000"
IMAGE_DIR = "code_images"

def get_ground_truth(filename):
    return os.path.splitext(filename)[0].strip().lower()

def normalize_text(text):
    text = text.strip().replace(" ", "").replace("-", "")
    return text.lower()

def test_image_accuracy(image_path, language="en"):
    url = f"{BASE_URL}/ocr/upload"
    
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/png")}
        data = {"language": language, "use_preprocessing": "True"}
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        full_text = result["full_text"]
        texts = [t["text"] for t in result["texts"]]
        confidence = max([t["confidence"] for t in result["texts"]]) if result["texts"] else 0
        
        return {
            "status": "success",
            "recognized_text": full_text,
            "recognized_texts": texts,
            "confidence": confidence,
            "text_count": len(result["texts"]),
            "time": elapsed,
            "details": result["texts"],
            "preprocessing": result.get("preprocessing_applied", [])
        }
    else:
        return {
            "status": "error",
            "error": response.text,
            "time": elapsed
        }

def calculate_match_score(ground_truth, recognized):
    gt = normalize_text(ground_truth)
    rec = normalize_text(recognized)
    
    if gt == rec:
        return 1.0
    
    if len(gt) == 0:
        return 0.0
    
    max_score = 0
    
    for i in range(max(1, len(rec) - len(gt) + 1)):
        for j in range(i + 1, min(len(rec) + 1, i + len(gt) + 3)):
            substring = rec[i:j]
            if substring == gt:
                max_score = 1.0
                break
        
        if max_score > 0:
            break
    
    if max_score < 1.0 and len(gt) > 0 and len(rec) > 0:
        matches = 0
        for char in gt:
            if char in rec:
                matches += 1
        max_score = matches / len(gt)
    
    return max_score

def main():
    print("=" * 80)
    print("OCR 验证码识别真实准确率测试")
    print("基准：图片文件名 = 验证码内容")
    print("=" * 80)
    
    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    if not image_files:
        print(f"No PNG images found in {IMAGE_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to test\n")
    
    results = []
    total_correct = 0
    total_partial = 0
    total_wrong = 0
    total_error = 0
    total_confidence = 0
    total_time = 0
    total_match_score = 0
    
    for img_path in sorted(image_files):
        ground_truth = get_ground_truth(img_path.name)
        print(f"Testing: {img_path.name} (expected: {ground_truth})...", end=" ", flush=True)
        
        result = test_image_accuracy(str(img_path))
        
        if result["status"] == "success":
            match_score = calculate_match_score(ground_truth, result["recognized_text"])
            result["ground_truth"] = ground_truth
            result["match_score"] = match_score
            
            total_confidence += result["confidence"]
            total_time += result["time"]
            total_match_score += match_score
            
            if match_score >= 0.99:
                status_str = "✓ CORRECT"
                total_correct += 1
            elif match_score >= 0.50:
                status_str = "⚠ PARTIAL"
                total_partial += 1
            else:
                status_str = "✗ WRONG"
                total_wrong += 1
            
            print(f"{status_str} | got: '{result['recognized_text']}' (match: {match_score:.2f}, conf: {result['confidence']:.4f})")
            
            results.append(result)
        else:
            total_error += 1
            total_time += result["time"]
            print(f"✗ Error")
            
            results.append({
                "image": img_path.name,
                "ground_truth": ground_truth,
                "status": "error",
                "error": result["error"],
                "time": result["time"],
                "match_score": 0
            })
    
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    print(f"\n总图片数: {len(image_files)}")
    print(f"完全正确: {total_correct} ({total_correct/len(image_files)*100:.1f}%)")
    print(f"部分正确: {total_partial} ({total_partial/len(image_files)*100:.1f}%)")
    print(f"完全错误: {total_wrong} ({total_wrong/len(image_files)*100:.1f}%)")
    print(f"识别失败: {total_error} ({total_error/len(image_files)*100:.1f}%)")
    print(f"\n真实准确率: {total_correct/len(image_files)*100:.1f}%")
    
    if len(image_files) - total_error > 0:
        avg_confidence = total_confidence / (len(image_files) - total_error)
        avg_match_score = total_match_score / (len(image_files) - total_error)
        avg_time = total_time / len(image_files)
        print(f"\n平均匹配度: {avg_match_score:.4f}")
        print(f"平均置信度: {avg_confidence:.4f}")
        print(f"平均识别时间: {avg_time:.2f}s")
        print(f"总耗时: {total_time:.2f}s")
    
    print("\n详细结果:")
    print("-" * 80)
    print(f"{'图片':20s} | {'期望':10s} | {'识别结果':15s} | {'匹配度':8s} | {'置信度':8s} | {'时间':6s}")
    print("-" * 80)
    for r in results:
        if r["status"] == "success":
            print(f"{r['image']:20s} | {r['ground_truth']:10s} | {r['recognized_text']:15s} | {r['match_score']:<8.2f} | {r['confidence']:<8.4f} | {r['time']:<6.2f}s")
        else:
            print(f"{r['image']:20s} | {r['ground_truth']:10s} | ERROR | - | - | {r['time']:<6.2f}s")
    
    with open("ocr_test_results_real.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n详细结果已保存到: ocr_test_results_real.json")

if __name__ == "__main__":
    main()
