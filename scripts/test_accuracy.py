#!/usr/bin/env python3
"""Run accuracy test against the OCR API."""

import requests
from pathlib import Path
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "http://localhost:8000"
DATA_DIR = Path(__file__).parent.parent / "data" / "samples"

def test_accuracy():
    imgs = sorted(DATA_DIR.glob("*.png"))
    if not imgs:
        print("No test images found in data/samples/")
        return
    
    total = len(imgs)
    print(f"Testing {total} captcha images...\n")

    ok = 0
    fail = 0
    
    for idx, img in enumerate(imgs, 1):
        exp = img.stem.lower()
        try:
            with open(img, "rb") as f:
                r = requests.post(f"{BASE_URL}/ocr/upload", 
                                files={"file": f}, 
                                data={"language": "en"}, 
                                timeout=15)
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

if __name__ == "__main__":
    test_accuracy()
