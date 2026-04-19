import json

with open("ocr_test_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

print("=" * 80)
print("OCR 验证码识别准确率测试报告")
print("=" * 80)

total = len(results)
success = sum(1 for r in results if r["status"] == "success")
failed = total - success

confs = [r["confidence"] for r in results if r["status"] == "success"]
times = [r["time"] for r in results]

avg_conf = sum(confs) / len(confs) if confs else 0
avg_time = sum(times) / len(times) if times else 0
min_conf = min(confs) if confs else 0
max_conf = max(confs) if confs else 0
total_time = sum(times)

print(f"\n总图片数: {total}")
print(f"成功识别: {success}")
print(f"识别失败: {failed}")
print(f"成功率: {success/total*100:.1f}%")
print(f"\n平均置信度: {avg_conf:.4f}")
print(f"最低置信度: {min_conf:.4f}")
print(f"最高置信度: {max_conf:.4f}")
print(f"平均识别时间: {avg_time:.2f}s")
print(f"总耗时: {total_time:.2f}s")

print("\n" + "-" * 80)
print("详细识别结果")
print("-" * 80)
print(f"{'图片':<20s} | {'识别结果':<20s} | {'置信度':<10s} | {'文本数':<8s} | {'时间':<8s}")
print("-" * 80)

for r in results:
    if r["status"] == "success":
        print(f"{r['image']:<20s} | {r['recognized_text']:<20s} | {r['confidence']:<10.4f} | {r['text_count']:<8d} | {r['time']:<8.2f}s")
    else:
        print(f"{r['image']:<20s} | ERROR | - | - | {r['time']:<8.2f}s")

print("\n" + "-" * 80)
print("置信度分布")
print("-" * 80)
excellent = sum(1 for c in confs if c >= 0.95)
good = sum(1 for c in confs if 0.90 <= c < 0.95)
medium = sum(1 for c in confs if 0.85 <= c < 0.90)
low = sum(1 for c in confs if c < 0.85)

print(f"优秀 (>=0.95): {excellent} ({excellent/total*100:.1f}%)")
print(f"良好 (0.90-0.95): {good} ({good/total*100:.1f}%)")
print(f"中等 (0.85-0.90): {medium} ({medium/total*100:.1f}%)")
print(f"较低 (<0.85): {low} ({low/total*100:.1f}%)")
