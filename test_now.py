import requests
from pathlib import Path
import os

imgs = sorted(Path("code_images").glob("*.png"))[:5]
print(f"Testing {len(imgs)} images...\n")

ok = 0
for img in imgs:
    exp = img.stem.lower()
    with open(img, "rb") as f:
        r = requests.post("http://localhost:8000/ocr/upload", files={"file": f}, data={"language": "en"})
    result = r.json()
    got = result.get("full_text", "").lower().replace(" ", "")
    m = "OK" if got == exp else "FAIL"
    if got == exp:
        ok += 1
    conf = result["texts"][0]["confidence"] if result.get("texts") else 0
    print(f"  {m}  {img.name:15s} | expected: {exp:8s} got: {got:10s} | conf: {conf:.3f}")

print(f"\nAccuracy: {ok}/{len(imgs)} = {ok/len(imgs)*100:.0f}%")
