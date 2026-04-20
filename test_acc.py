import requests
import os
from pathlib import Path
import io
from PIL import Image

BASE = "http://localhost:8000"
images = sorted(Path("code_images").glob("*.png"))[:15]

print(f"测试前 {len(images)} 张图片的识别准确率...\n")

correct = 0
total = 0

for img in images:
    expected = os.path.splitext(img.name)[0].lower()
    try:
        with open(img, "rb") as f:
            img_data = f.read()
            pil_img = Image.open(io.BytesIO(img_data))
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            files = {"file": (img.name, img_bytes, "image/png")}
            data = {"language": "en"}
            r = requests.post(f"{BASE}/ocr/upload", files=files, data=data, timeout=30)
        
        if r.status_code == 200:
            got = r.json()["full_text"].strip().replace(" ", "").lower()
            match = (got == expected)
            status = "OK" if match else "FAIL"
            print(f"  {status}  {img.name:15s} -> 期望: {expected:8s} 识别: {got:10s}")
            if match:
                correct += 1
            total += 1
        else:
            print(f"  ERROR {img.name:15s} -> HTTP {r.status_code}: {r.text[:80]}")
    except Exception as e:
        print(f"  ERROR {img.name:15s} -> {str(e)[:40]}")

print(f"\n{'='*60}")
if total > 0:
    print(f"准确率: {correct}/{total} = {correct/total*100:.1f}%")
else:
    print("无有效结果")
