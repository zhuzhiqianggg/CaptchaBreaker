#!/usr/bin/env python3
"""
Simple test for v6.0.0
"""

import sys
import time
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data" / "samples"

print("Testing v6.0.0 preprocessing strategies...")

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
    total_start = time.time()
    
    for i, img_path in enumerate(imgs[:5], 1):
        print(f"\n[{i}/{len(imgs[:5])}] Testing: {img_path.name}")
        print(f"Expected: {img_path.stem.lower()}")
        
        image = Image.open(img_path)
        
        strategies = [
            ("v1:Light", preprocess_image_v1),
            ("v2:Medium", preprocess_image_v2),
            ("v3:Binarize", preprocess_image_v3),
            ("v4:Denoise", preprocess_image_v4),
        ]
        
        all_texts = []
        best_text = ""
        best_confidence = -1
        best_strategy = ""
        
        for name, strategy in strategies:
            processed, steps = strategy(image)
            
            temp_path = Path(__file__).parent.parent / "temp" / f"test_{img_path.stem}_{name}.png"
            temp_path.parent.mkdir(exist_ok=True)
            processed.save(temp_path)
            
            result = ocr.ocr(str(temp_path))
            
            texts, full_text = parse_ocr_result(result)
            all_texts.append(full_text)
            
            avg_conf = 0.0
            if texts:
                avg_conf = sum(t.confidence for t in texts) / len(texts)
            
            print(f"  {name:15s}: \"{full_text}\" (conf: {avg_conf:.2f})")
            
            if avg_conf > best_confidence:
                best_confidence = avg_conf
                best_text = full_text
                best_strategy = name
        
        voted = vote_results(all_texts)
        final = post_process_text(voted)
        
        expected = img_path.stem.lower().strip()
        match = (final == expected) or (best_text.lower() == expected)
        
        print(f"  Voted: \"{voted}\" -> Final: \"{final}\"")
        print(f"  Best strategy: {best_strategy} (conf: {best_confidence:.2f})")
        print(f"  Match: {'[OK]' if match else '[FAIL]'}")
        
        results.append(match)
    
    elapsed = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"Results: {sum(results)}/{len(results)} correct")
    print(f"Accuracy: {sum(results)/len(results)*100:.1f}%")
    print(f"Time: {elapsed:.1f}s")
    print(f"{'='*80}\n")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
