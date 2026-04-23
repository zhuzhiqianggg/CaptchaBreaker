#!/usr/bin/env python3
"""
Full test for v6.0.0
"""

import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "samples"

print("Testing v6.0.0 - Full test on all 31 images...")

try:
    from app.main import (
        preprocess_image_v1,
        preprocess_image_v2, 
        preprocess_image_v3,
        preprocess_image_v4,
        post_process_text,
        vote_results,
        parse_ocr_result
    )
    from paddleocr import PaddleOCR
    
    print("Loading PaddleOCR model...")
    ocr = PaddleOCR(lang='en')
    print("Model loaded!\n")
    
    imgs = sorted(DATA_DIR.glob("*.png"))
    
    if not imgs:
        print("No test images found!")
        sys.exit(1)
    
    print(f"Testing {len(imgs)} images...\n")
    print("="*80)
    
    results = []
    failed = []
    total_start = time.time()
    
    for i, img_path in enumerate(imgs, 1):
        print(f"\n[{i}/{len(imgs)}] {img_path.name}", end="")
        
        image = Image.open(img_path)
        
        strategies = [
            preprocess_image_v1,
            preprocess_image_v2,
            preprocess_image_v3,
            preprocess_image_v4,
        ]
        
        all_texts = []
        best_text = ""
        best_confidence = -1
        
        for strategy in strategies:
            processed, steps = strategy(image)
            
            temp_path = Path(__file__).parent.parent / "temp" / f"test_{img_path.stem}_{strategy.__name__}.png"
            temp_path.parent.mkdir(exist_ok=True)
            processed.save(temp_path)
            
            result = ocr.ocr(str(temp_path))
            texts, full_text = parse_ocr_result(result)
            all_texts.append(full_text)
            
            avg_conf = 0.0
            if texts:
                avg_conf = sum(t.confidence for t in texts) / len(texts)
            
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_text = full_text
        
        voted = vote_results(all_texts)
        final = post_process_text(voted)
        
        expected = img_path.stem.lower().strip()
        
        if final == expected:
            print(f" [OK] Expected: {expected:10s} Got: {final:10s}")
            results.append(True)
        elif best_text.lower() == expected:
            print(f" [OK] Expected: {expected:10s} Got: {final:10s} (best: {best_text})")
            results.append(True)
        else:
            print(f" [FAIL] Expected: {expected:10s} Got: {final:10s} (best: {best_text})")
            results.append(False)
            failed.append({
                "file": img_path.name,
                "expected": expected,
                "got": final,
                "best": best_text
            })
    
    elapsed = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"Results: {sum(results)}/{len(results)} correct")
    print(f"Accuracy: {sum(results)/len(results)*100:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(imgs):.2f}s per image)")
    print(f"{'='*80}\n")
    
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f['file']:20s} Expected: {f['expected']:10s} Got: {f['got']:10s} Best: {f['best']}")
        print()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
