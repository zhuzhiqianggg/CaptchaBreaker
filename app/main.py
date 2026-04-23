import os
import uuid
import base64
import re
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
    title="CaptchaBreaker API",
    description="Local OCR service for captcha recognition using PaddleOCR",
    version="5.0.0",
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


def preprocess_image(image: Image.Image) -> Tuple[Image.Image, List[str]]:
    steps = []
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if width < 200 or height < 60:
        scale = max(200/width, 60/height)
        new_w = int(width * scale)
        new_h = int(height * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
        steps.append(f"Upscale from {width}x{height} to {new_w}x{new_h}")
    
    img_gray = image.convert("L")
    enhancer = ImageEnhance.Contrast(img_gray)
    img_enhanced = enhancer.enhance(1.5)
    steps.append("Light contrast enhancement")
    
    return img_enhanced.convert("RGB"), steps


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

    image_id = uuid.uuid4().hex
    file_path = None

    try:
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".png"
        if not suffix or len(suffix) < 2:
            suffix = ".png"
        file_path = save_upload_file(file, suffix)

        image = Image.open(file_path)
        processed_image, steps = preprocess_image(image)
        
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

        processed_image, steps = preprocess_image(image)

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

        processed_image, steps = preprocess_image(image)

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
        "service": "CaptchaBreaker OCR API",
        "version": "5.0.0",
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
