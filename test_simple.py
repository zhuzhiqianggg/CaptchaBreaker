import requests, os
from pathlib import Path

imgs = list(Path("code_images").glob("*.png"))[:5]
print(f"测试 {len(imgs)} 张图片\n")

ok = 0
for img in imgs:
    exp = img.stem.lower()
    with open(img, "rb") as f:
        r = requests.post("http://localhost:8000/ocr/upload", 
                         files={"file": f}, data={"language": "en"})
    got = r.json().get("full_text","").lower().replace(" ","")
    m = "OK" if got==exp else "FAIL"
    if got==exp: ok+=1
    print(f"{m}  {img.name:15s} | 期望:{exp:8s} 识别:{got:10s}")

print(f"\n准确率: {ok}/{len(imgs)} = {ok/len(imgs)*100:.0f}%")
