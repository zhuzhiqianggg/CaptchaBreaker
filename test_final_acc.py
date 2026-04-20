import requests
from pathlib import Path
import os, sys

imgs = sorted(Path("code_images").glob("*.png"))
total = len(imgs)
print(f"Testing {total} images...\n")

ok = 0
fail = 0
idx = 0

for img in imgs:
    idx += 1
    exp = img.stem.lower()
    try:
        with open(img, "rb") as f:
            r = requests.post("http://localhost:8000/ocr/upload", files={"file": f}, data={"language": "en"}, timeout=15)
        result = r.json()
        got = result.get("full_text", "").lower().replace(" ", "")
        match = (got == exp)
        if match: ok += 1
        else: fail += 1
        conf = result["texts"][0]["confidence"] if result.get("texts") else 0
        tag = "OK" if match else "FAIL"
        print(f"  [{idx:2d}/{total}] {tag:4s} {img.name:15s} | exp: {exp:8s} got: {got:10s} conf: {conf:.3f}")
        sys.stdout.flush()
    except Exception as e:
        fail += 1
        print(f"  [{idx:2d}/{total}] ERR  {img.name:15s} | {str(e)[:30]}")
        sys.stdout.flush()

print(f"\n{'='*70}")
print(f"Accuracy: {ok}/{total} = {ok/total*100:.1f}%")
print(f"Correct: {ok} | Wrong: {fail}")
