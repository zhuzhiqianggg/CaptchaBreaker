#!/usr/bin/env python3
"""
Test v6.0.0 accuracy with multi-strategy voting
"""

import os
import sys
import time
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_URL = "http://127.0.0.1:8000"
DATA_DIR = Path(__file__).parent.parent / "data" / "samples"


def test_image(filepath: Path) -> dict:
    expected = filepath.stem.lower().strip()
    
    with open(filepath, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/ocr/upload",
            files={"file": (filepath.name, f, "image/png")},
            data={"language": "general"},
            timeout=30
        )
    
    if resp.status_code != 200:
        return {"file": filepath.name, "expected": expected, "got": "ERROR", "match": False}
    
    result = resp.json()
    got = result.get("full_text", "").lower().replace(" ", "").strip()
    
    return {
        "file": filepath.name,
        "expected": expected,
        "got": got,
        "match": (got == expected),
        "preprocessing": result.get("preprocessing_applied", [])
    }


def main():
    imgs = sorted(DATA_DIR.glob("*.png"))
    
    if not imgs:
        print("No test images found")
        return
    
    print(f"\n{'='*60}")
    print(f"CaptchaBreaker v6.0.0 Accuracy Test")
    print(f"{'='*60}\n")
    
    results = []
    start = time.time()
    
    for i, img in enumerate(imgs, 1):
        r = test_image(img)
        results.append(r)
        
        status = "[OK]" if r["match"] else "[FAIL]"
        print(f"{i:2d}. {status} {r['file']:30s} Expected: {r['expected']:10s} Got: {r['got'][:15]:15s}")
        
        if not r["match"]:
            print(f"     Preprocessing: {', '.join(r.get('preprocessing', [])[:3])}")
    
    elapsed = time.time() - start
    
    matched = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = matched / total * 100 if total > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Results: {matched}/{total} correct")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/total:.2f}s per image)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
