import os
import uuid
import base64
import re
import json
import statistics
import requests as http_requests
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
from collections import Counter

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageFilter

from paddleocr import PaddleOCR

ocr_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ocr_models["general"] = PaddleOCR(lang='ch')
    ocr_models["en"] = PaddleOCR(lang='en')
    print("PaddleOCR models loaded successfully")
    
    warmup = os.getenv("MODEL_WARMUP", "true").lower() == "true"
    if warmup:
        try:
            print("Warming up OCR models...")
            import numpy as np
            warmup_img = Image.fromarray(np.random.randint(0, 255, (40, 100, 3), dtype=np.uint8))
            temp_dir = os.path.join(os.path.dirname(__file__), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            warmup_path = os.path.join(temp_dir, "warmup.png")
            warmup_img.save(warmup_path)
            for model in ocr_models.values():
                model.ocr(warmup_path)
            cleanup_file(warmup_path)
            print("Model warmup completed")
        except Exception as e:
            print(f"Warmup skipped: {e}")
    
    yield
    ocr_models.clear()
    print("PaddleOCR models unloaded")


app = FastAPI(
    title="CaptchaBreaker API",
    description="Local OCR service for captcha recognition using PaddleOCR",
    version="7.0.0",
    lifespan=lifespan
)


class OCRResult(BaseModel):
    text: str
    confidence: float
    bounding_box: Optional[Dict[str, float]] = None


class OCRResponse(BaseModel):
    success: bool
    image_id: str
    texts: List[OCRResult]
    full_text: str
    language: str
    message: str
    preprocessing_applied: List[str] = []


class URLOCRRequest(BaseModel):
    image_url: str
    language: str = "general"


TEMP_DIR = os.path.join(os.path.dirname(__file__), "temp")
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "collected")


def save_upload_file(upload_file: UploadFile, suffix: str = ".png") -> str:
    os.makedirs(TEMP_DIR, exist_ok=True)
    file_path = os.path.join(TEMP_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(file_path, "wb") as f:
        content = upload_file.file.read()
        f.write(content)
    return file_path


def cleanup_file(file_path: str):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception:
        pass


def get_train_dirs():
    return {
        "original": os.path.join(TRAIN_DATA_DIR, "original"),
        "v1": os.path.join(TRAIN_DATA_DIR, "v1_light"),
        "v2": os.path.join(TRAIN_DATA_DIR, "v2_medium"),
        "v3": os.path.join(TRAIN_DATA_DIR, "v3_binary"),
        "v4": os.path.join(TRAIN_DATA_DIR, "v4_denoise"),
        "metadata": os.path.join(TRAIN_DATA_DIR, "metadata"),
    }


def save_training_data(
    image_id: str,
    original_image: Image.Image,
    processed_images: Dict[str, Tuple[Image.Image, List[str]]],
    ocr_results: Dict[str, str],
    final_text: str,
    corrected_text: str = None
):
    try:
        train_dirs = get_train_dirs()
        
        for dir_path in train_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        original_path = os.path.join(train_dirs["original"], f"{image_id}.png")
        original_image.save(original_path)
        
        metadata = {
            "image_id": image_id,
            "final_text": final_text,
            "corrected_text": corrected_text,
            "strategies": {}
        }
        
        for strategy_name, (processed_img, steps) in processed_images.items():
            dir_key = strategy_name.replace("preprocess_image_", "")
            save_path = os.path.join(train_dirs[dir_key], f"{image_id}.png")
            processed_img.save(save_path)
            
            metadata["strategies"][dir_key] = {
                "ocr_result": ocr_results.get(strategy_name, ""),
                "preprocessing_steps": steps
            }
        
        metadata_path = os.path.join(train_dirs["metadata"], f"{image_id}.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return True
    except Exception as e:
        print(f"Failed to save training data: {e}")
        return False


CHAR_CONFUSION_MAP = {
    '0': ['O', 'o', 'Q'],
    'O': ['0', 'o', 'Q'],
    'o': ['0', 'O', 'Q'],
    '1': ['I', 'l', 'i', 'L'],
    'I': ['1', 'l', 'i', 'L'],
    'l': ['1', 'I', 'i', 'L'],
    'i': ['1', 'I', 'l', 'L'],
    'L': ['1', 'I', 'l', 'i'],
    '2': ['Z', 'z'],
    'Z': ['2', 'z'],
    '5': ['S', 's'],
    'S': ['5', 's'],
    '6': ['b', 'G'],
    'b': ['6', 'G'],
    '8': ['B'],
    'B': ['8'],
    '9': ['g', 'q'],
    'g': ['9', 'q'],
    'q': ['9', 'g'],
    'm': ['rn', 'nn'],
    'w': ['vv'],
    'W': ['VV'],
    'vv': ['w', 'W'],
    'VV': ['W', 'w'],
    '7': ['T', 't'],
    'T': ['7'],
    't': ['7'],
    'd': ['cl', 'd'],
    'D': ['D'],
    'u': ['u'],
    'U': ['U'],
    'v': ['v', 'V'],
    'V': ['v', 'V'],
    'r': ['r'],
    'R': ['R'],
    'n': ['n'],
    'N': ['N'],
    'f': ['f'],
    'F': ['F'],
    '3': ['3'],
    '4': ['4'],
    'A': ['A', 'a'],
    'a': ['A', 'a'],
    'C': ['C'],
    'c': ['c'],
    'E': ['E'],
    'e': ['e'],
    'H': ['H'],
    'h': ['h'],
    'J': ['J'],
    'j': ['j'],
    'K': ['K'],
    'k': ['k'],
    'M': ['M'],
    'P': ['P'],
    'p': ['p'],
    's': ['s'],
    'X': ['X'],
    'x': ['x'],
    'Y': ['Y'],
    'y': ['y'],
}

COMMON_PATTERNS = {
    'rn': 'm',
    'cl': 'd',
    'vv': 'w',
    'VV': 'W',
    'll': 'w',
    'II': 'w',
}

CAPTCHA_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

PREPROCESS_STRATEGIES = [
    "preprocess_image_v1",
    "preprocess_image_v2",
    "preprocess_image_v3",
    "preprocess_image_v4",
]


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def chars_similar(c1: str, c2: str) -> bool:
    if c1 == c2:
        return True
    
    c1_lower = c1.lower()
    c2_lower = c2.lower()
    
    if c1_lower == c2_lower:
        return True
    
    if c2 in CHAR_CONFUSION_MAP.get(c1, []):
        return True
    
    if c1 in CHAR_CONFUSION_MAP.get(c2, []):
        return True
    
    return False


def is_valid_captcha(text: str) -> bool:
    if not text:
        return False
    
    if len(text) < 3 or len(text) > 10:
        return False
    
    for char in text:
        if char not in CAPTCHA_CHARS:
            return False
    
    return True


def generate_candidates(text: str) -> List[str]:
    candidates = [text]
    
    for i, char in enumerate(text):
        if char in CHAR_CONFUSION_MAP:
            for replacement in CHAR_CONFUSION_MAP[char]:
                if len(replacement) == 1:
                    candidate = text[:i] + replacement + text[i+1:]
                    candidates.append(candidate)
        
        if char.islower():
            upper_char = char.upper()
            if upper_char != char:
                candidate = text[:i] + upper_char + text[i+1:]
                candidates.append(candidate)
        
        if char.isupper():
            lower_char = char.lower()
            if lower_char != char and lower_char in CAPTCHA_CHARS:
                candidate = text[:i] + lower_char + text[i+1:]
                candidates.append(candidate)
    
    for pattern, replacement in COMMON_PATTERNS.items():
        if pattern in text:
            candidate = text.replace(pattern, replacement)
            candidates.append(candidate)
    
    return candidates


def smart_correct(text: str, expected_length: int = 4) -> str:
    if not text:
        return text
    
    text = text.strip()
    
    if len(text) == expected_length and is_valid_captcha(text):
        return text
    
    candidates = generate_candidates(text)
    
    if len(text) > expected_length:
        for i in range(len(text) - expected_length + 1):
            substring = text[i:i+expected_length]
            if is_valid_captcha(substring):
                candidates.append(substring)
                for candidate in generate_candidates(substring):
                    candidates.append(candidate)
    
    if len(text) < expected_length:
        common_prefixes = [c for c in CAPTCHA_CHARS if c.isdigit() or c.isupper()]
        
        for prefix in common_prefixes:
            candidate = prefix + text
            if len(candidate) == expected_length and is_valid_captcha(candidate):
                candidates.append(candidate)
        
        for position in range(len(text) + 1):
            for char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G']:
                candidate = text[:position] + char + text[position:]
                if len(candidate) == expected_length and is_valid_captcha(candidate):
                    candidates.append(candidate)
    
    best_candidate = text
    best_score = -1
    
    for candidate in candidates:
        if not is_valid_captcha(candidate):
            continue
        
        if len(candidate) != expected_length:
            continue
        
        distance = levenshtein_distance(candidate.lower(), text.lower())
        similarity = 1.0 - (distance / max(len(candidate), len(text)))
        
        char_match_score = sum(
            1 for c1, c2 in zip(candidate.lower(), text.lower()) 
            if chars_similar(c1, c2)
        ) / max(len(candidate), len(text))
        
        case_bonus = 0.0
        if len(candidate) == len(text):
            case_matches = sum(
                1 for c1, c2 in zip(candidate, text)
                if (c1.isupper() == c2.isupper()) or (c1.lower() == c2.lower())
            )
            case_bonus = (case_matches / len(candidate)) * 0.1
        
        score = similarity * 0.4 + char_match_score * 0.6 + case_bonus
        
        if score > best_score:
            best_score = score
            best_candidate = candidate
    
    return best_candidate


def fix_case_by_confidence(text: str, ocr_texts: List[str]) -> str:
    if not text or not ocr_texts:
        return text
    
    if len(text) != 4:
        return text
    
    case_votes = [0, 0, 0, 0]
    
    for ocr_text in ocr_texts:
        if len(ocr_text) != 4:
            continue
        
        for i, char in enumerate(ocr_text):
            if char.isupper():
                case_votes[i] += 1
            elif char.islower():
                case_votes[i] -= 1
    
    fixed_text = list(text)
    for i in range(min(4, len(text))):
        if text[i].isalpha():
            if case_votes[i] > 0:
                fixed_text[i] = text[i].upper()
            elif case_votes[i] < 0:
                fixed_text[i] = text[i].lower()
    
    return "".join(fixed_text)


def post_process_text(text: str) -> str:
    corrected = text
    for pattern, replacement in COMMON_PATTERNS.items():
        corrected = corrected.replace(pattern, replacement)
    return corrected


def preprocess_image_v1(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    steps = ["v1:Light"]
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if width < 200 or height < 60:
        scale = max(200/width, 60/height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        steps.append(f"Upscale to {new_w}x{new_h}")
    
    img_gray = image.convert("L")
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(1.5)
    steps.append("Contrast+1.5x")
    
    return img_enhanced.convert("RGB"), steps


def preprocess_image_v2(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    steps = ["v2:Medium"]
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if width < 200 or height < 60:
        scale = max(200/width, 60/height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        steps.append(f"Upscale to {new_w}x{new_h}")
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    steps.append("Contrast+2.0x")
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.5)
    steps.append("Sharpness+1.5x")
    
    return image, steps


def preprocess_image_v3(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    steps = ["v3:Binarize"]
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if width < 200 or height < 60:
        scale = max(200/width, 60/height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        steps.append(f"Upscale to {new_w}x{new_h}")
    
    img_gray = image.convert("L")
    
    threshold = 128
    img_bin = img_gray.point(lambda x: 255 if x > threshold else 0, '1')
    steps.append(f"Binary threshold={threshold}")
    
    return img_bin.convert("RGB"), steps


def preprocess_image_v4(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    steps = ["v4:Denoise"]
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if width < 200 or height < 60:
        scale = max(200/width, 60/height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        steps.append(f"Upscale to {new_w}x{new_h}")
    
    image = image.filter(ImageFilter.MedianFilter(size=3))
    steps.append("Median filter")
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    steps.append("Contrast+1.3x")
    
    return image, steps


def parse_ocr_result(result) -> Tuple[List[OCRResult], str]:
    texts = []
    full_text_parts = []

    if result and result[0]:
        ocr_data = result[0]
        
        if hasattr(ocr_data, 'rec_texts'):
            rec_texts = ocr_data.rec_texts
            rec_scores = ocr_data.rec_scores
            rec_polys = ocr_data.rec_polys if hasattr(ocr_data, 'rec_polys') else []
        elif isinstance(ocr_data, dict):
            rec_texts = ocr_data.get('rec_texts', [])
            rec_scores = ocr_data.get('rec_scores', [])
            rec_polys = ocr_data.get('rec_polys', [])
        else:
            rec_texts = []
            rec_scores = []
            rec_polys = []

        for i, recognized_text in enumerate(rec_texts):
            if recognized_text and recognized_text.strip():
                confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.0
                
                bbox = None
                if i < len(rec_polys):
                    poly = rec_polys[i]
                    if hasattr(poly, 'tolist'):
                        poly = poly.tolist()
                    if poly and len(poly) == 4:
                        x_coords = [p[0] for p in poly]
                        y_coords = [p[1] for p in poly]
                        bbox = {
                            "x_min": float(min(x_coords)),
                            "y_min": float(min(y_coords)),
                            "x_max": float(max(x_coords)),
                            "y_max": float(max(y_coords))
                        }

                texts.append(OCRResult(
                    text=recognized_text.strip(),
                    confidence=confidence,
                    bounding_box=bbox
                ))
                full_text_parts.append(recognized_text.strip())

    full_text = "".join(full_text_parts)
    full_text = re.sub(r'\s+', '', full_text)

    return texts, full_text


def vote_results(results: List[str]) -> str:
    if not results:
        return ""
    
    if len(results) == 1:
        return results[0]
    
    cleaned_results = []
    original_results = {}
    for r in results:
        cleaned = re.sub(r'\s+', '', r).lower()
        if cleaned:
            cleaned_results.append(cleaned)
            if cleaned not in original_results:
                original_results[cleaned] = r
    
    if not cleaned_results:
        return ""
    
    counter = Counter(cleaned_results)
    most_common = counter.most_common()
    
    if most_common[0][1] > 1:
        return original_results[most_common[0][0]]
    
    lengths = [len(r) for r in cleaned_results]
    avg_length = statistics.mean(lengths) if lengths else 0
    closest = min(cleaned_results, key=lambda x: abs(len(x) - avg_length))
    
    return original_results[closest]


def process_ocr_pipeline(image: Image.Image, language: str, image_id: str, file_path: str = None) -> OCRResponse:
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    strategy_funcs = [
        preprocess_image_v1,
        preprocess_image_v2,
        preprocess_image_v3,
        preprocess_image_v4,
    ]
    
    all_full_texts = []
    best_texts = []
    best_steps = []
    best_confidence = -1
    processed_images = {}
    ocr_results = {}
    
    for strategy in strategy_funcs:
        processed_image, steps = strategy(image)
        
        temp_path = os.path.join(TEMP_DIR, f"{image_id}_{strategy.__name__}.png")
        processed_image.save(temp_path)
        
        result = ocr_models[language].predict(temp_path)
        
        texts, full_text = parse_ocr_result(result)
        all_full_texts.append(full_text)
        processed_images[strategy.__name__] = (processed_image.copy(), steps)
        ocr_results[strategy.__name__] = full_text
        
        avg_confidence = 0.0
        if texts:
            avg_confidence = sum(t.confidence for t in texts) / len(texts)
        
        if avg_confidence > best_confidence:
            best_confidence = avg_confidence
            best_texts = texts
            best_steps = steps
        
        cleanup_file(temp_path)
    
    voted_full_text = vote_results(all_full_texts)
    
    final_texts = best_texts
    final_steps = best_steps + ["Multi-strategy voting applied", "Smart correction enabled"]
    
    corrected = None
    if voted_full_text:
        corrected = smart_correct(voted_full_text, expected_length=4)
        final_full_text = corrected
        
        fixed_text = fix_case_by_confidence(final_full_text, all_full_texts)
        if fixed_text != final_full_text:
            final_full_text = fixed_text
            final_steps.append("Case correction applied")
        
        if corrected != voted_full_text:
            final_texts = [OCRResult(
                text=final_full_text,
                confidence=best_confidence,
                bounding_box=None
            )]
    else:
        final_full_text = "".join([t.text for t in best_texts])
        final_full_text = re.sub(r'\s+', '', final_full_text)
        final_full_text = smart_correct(final_full_text, expected_length=4)
        final_full_text = fix_case_by_confidence(final_full_text, all_full_texts)
        final_texts = best_texts
    
    save_training_data(
        image_id=image_id,
        original_image=image,
        processed_images=processed_images,
        ocr_results=ocr_results,
        final_text=voted_full_text if voted_full_text else final_full_text,
        corrected_text=corrected if voted_full_text and corrected != voted_full_text else None
    )
    
    return OCRResponse(
        success=True,
        image_id=image_id,
        texts=final_texts,
        full_text=final_full_text,
        language=language,
        message=f"Successfully recognized {len(final_texts)} text(s) with multi-strategy voting",
        preprocessing_applied=final_steps
    )


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(...),
    language: str = Form(default="general")
):
    if language not in ocr_models:
        language = "general"

    image_id = uuid.uuid4().hex
    file_path = None

    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".png"
        if not suffix or len(suffix) < 2:
            suffix = ".png"
        file_path = save_upload_file(file, suffix)

        image = Image.open(file_path)
        
        return process_ocr_pipeline(image, language, image_id, file_path)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    finally:
        if file_path:
            cleanup_file(file_path)


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(
    image_data: str = Form(...),
    language: str = Form(default="general")
):
    if language not in ocr_models:
        language = "general"

    image_id = uuid.uuid4().hex

    try:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return process_ocr_pipeline(image, language, image_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_url(request: URLOCRRequest):
    language = request.language if request.language in ocr_models else "general"
    image_id = uuid.uuid4().hex

    try:
        response = http_requests.get(request.image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        return process_ocr_pipeline(image, language, image_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "service": "CaptchaBreaker OCR API",
        "version": "7.0.0",
        "endpoints": {
            "POST /ocr/upload": "Upload image file",
            "POST /ocr/base64": "Base64 image data",
            "POST /ocr/url": "Image URL",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "ocr_models_loaded": len(ocr_models) > 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
