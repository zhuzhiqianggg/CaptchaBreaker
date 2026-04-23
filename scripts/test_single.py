#!/usr/bin/env python3
"""Quick test for a single captcha image."""

import requests
from pathlib import Path
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_single(image_path: str):
    img = Path(image_path)
    if not img.exists():
        print(f"File not found: {img}")
        return
    
    exp = img.stem.lower()
    print(f"Testing: {img.name}")
    print(f"Expected: {exp}\n")
    
    with open(img, "rb") as f:
        r = requests.post("http://localhost:8000/ocr/upload", 
                        files={"file": f}, 
                        data={"language": "en"},
                        timeout=15)
    
    if r.status_code == 200:
        result = r.json()
        got = result.get("full_text", "").lower().replace(" ", "")
        match = "MATCH" if got == exp else "MISMATCH"
        
        print(f"Result: {got}")
        print(f"Status: {match}")
        print(f"Confidence: {result['texts'][0]['confidence'] if result.get('texts') else 0:.3f}")
        print(f"Preprocessing: {result.get('preprocessing_applied', [])}")
    else:
        print(f"Error: HTTP {r.status_code}")
        print(r.text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single(sys.argv[1])
    else:
        print("Usage: python test_single.py <image_path>")
        print("Example: python test_single.py data/samples/2zrw.png")
