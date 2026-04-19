import os
import uuid
import base64
import re
import cv2
import numpy as np
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageEnhance

from paddleocr import PaddleOCR

ocr_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ocr_models["general"] = PaddleOCR(lang='ch')
    ocr_models["en"] = PaddleOCR(lang='en')
    print("PaddleOCR models loaded successfully")
    yield
    ocr_models.clear()
    print("PaddleOCR models unloaded")

app = FastAPI(
    title="OCR Image Recognition API",
    description="Local OCR service for image captcha recognition using PaddleOCR",
    version="4.0.0",
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

def save_upload_file(upload_file: UploadFile, suffix: str = ".png") -> str:
    temp_dir = os.path.join(os.path.dirname(__file__), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{uuid.uuid4().hex}{suffix}")
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

def preprocess_captcha(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    steps = []
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    original_width, original_height = image.width, image.height
    steps.append(f"Original size: {original_width}x{original_height}")
    
    img_gray = image.convert("L")
    
    enhancer = ImageEnhance.Contrast(img_gray)
    img_gray = enhancer.enhance(2.0)
    steps.append("Contrast enhancement (2.0x)")
    
    enhancer = ImageEnhance.Brightness(img_gray)
    img_gray = enhancer.enhance(1.0)
    steps.append("Brightness enhancement (1.0x)")
    
    img_array = np.array(img_gray)
    
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    steps.append("Otsu thresholding")
    
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    steps.append("Morphological operations (close + open)")
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    min_area = 15
    cleaned = np.zeros_like(binary)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 0
        else:
            cleaned[labels == i] = 255
    
    steps.append(f"Connected components filtering (min_area={min_area})")
    
    cleaned_pil = Image.fromarray(cleaned).convert("RGB")
    
    scale_factor = 3
    new_width = original_width * scale_factor
    new_height = original_height * scale_factor
    cleaned_pil = cleaned_pil.resize((new_width, new_height), Image.LANCZOS)
    steps.append(f"Upscale {scale_factor}x to {new_width}x{new_height}")
    
    return cleaned_pil, steps

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

@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(
    file: UploadFile = File(...),
    language: str = Form(default="general")
):
    if language not in ocr_models:
        language = "general"

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    supported_formats = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp']
    if file.content_type not in supported_formats:
        raise HTTPException(status_code=400, detail=f"Unsupported image format. Supported: {supported_formats}")

    image_id = uuid.uuid4().hex
    file_path = None

    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".png"
        if not suffix or len(suffix) < 2:
            suffix = ".png"
        file_path = save_upload_file(file, suffix)

        image = Image.open(file_path)
        processed_image, steps = preprocess_captcha(image)
        
        processed_path = file_path.replace(suffix, "_processed.png")
        processed_image.save(processed_path)
        
        result = ocr_models[language].ocr(processed_path)
        
        if os.path.exists(processed_path):
            os.remove(processed_path)

        texts, full_text = parse_ocr_result(result)

        return OCRResponse(
            success=True,
            image_id=image_id,
            texts=texts,
            full_text=full_text,
            language=language,
            message=f"Successfully recognized {len(texts)} text(s)",
            preprocessing_applied=steps
        )

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

        processed_image, steps = preprocess_captcha(image)

        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{image_id}.png")
        processed_image.save(file_path)

        result = ocr_models[language].ocr(file_path)

        texts, full_text = parse_ocr_result(result)

        return OCRResponse(
            success=True,
            image_id=image_id,
            texts=texts,
            full_text=full_text,
            language=language,
            message=f"Successfully recognized {len(texts)} text(s)",
            preprocessing_applied=steps
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    finally:
        if file_path:
            cleanup_file(file_path)

@app.post("/ocr/url", response_model=OCRResponse)
async def ocr_url(request: URLOCRRequest):
    import requests as req
    
    language = request.language if request.language in ocr_models else "general"
    
    image_id = uuid.uuid4().hex

    try:
        response = req.get(request.image_url, timeout=30)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        
        if image.mode != "RGB":
            image = image.convert("RGB")

        processed_image, steps = preprocess_captcha(image)

        temp_dir = os.path.join(os.path.dirname(__file__), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f"{image_id}.png")
        processed_image.save(file_path)

        result = ocr_models[language].ocr(file_path)

        texts, full_text = parse_ocr_result(result)

        return OCRResponse(
            success=True,
            image_id=image_id,
            texts=texts,
            full_text=full_text,
            language=language,
            message=f"Successfully recognized {len(texts)} text(s)",
            preprocessing_applied=steps
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    finally:
        if file_path:
            cleanup_file(file_path)

@app.get("/")
async def root():
    return {
        "service": "OCR Image Recognition API",
        "version": "4.0.0",
        "features": {
            "noise_removal": "CV2-based noise and interference line removal",
            "connected_components": "Remove small noise dots using connected components analysis",
            "adaptive_thresholding": "Otsu thresholding for optimal binarization",
            "image_upscaling": "3x upscale for better character recognition"
        },
        "endpoints": {
            "POST /ocr/upload": "Upload an image file for OCR recognition",
            "POST /ocr/base64": "Send base64 encoded image for OCR recognition",
            "POST /ocr/url": "Send image URL for OCR recognition",
            "GET /health": "Health check endpoint"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ocr_models_loaded": len(ocr_models) > 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
