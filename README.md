# CaptchaBreaker

> Local OCR Captcha Recognition Service powered by PaddleOCR

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.4-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

CaptchaBreaker is a **local, privacy-first** OCR service designed for automated captcha recognition. It combines the powerful PaddleOCR deep learning model with intelligent image preprocessing to achieve reliable captcha recognition without sending data to external services.

## Features

- **3 Recognition Methods**: File upload, Base64 data, and URL
- **Smart Preprocessing**: Automatic grayscale conversion, contrast enhancement, and upscaling
- **JSON Response**: Structured output with recognized text, confidence scores, and bounding boxes
- **Multi-language Support**: Chinese and English recognition models
- **Local Processing**: All data stays on your machine - zero privacy concerns
- **RESTful API**: FastAPI-based server with health check and auto-generated docs

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zhuzhiqianggg/CaptchaBreaker.git
cd CaptchaBreaker

# Install dependencies
pip install -r requirements.txt

# Start the server
python run.py
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Recognize a captcha image
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@data/samples/2zrw.png" \
  -F "language=en"
```

### Run Tests

```bash
# Run accuracy test on sample images
python scripts/test_accuracy.py

# Test a single image
python scripts/test_single.py data/samples/2zrw.png
```

## Project Structure

```
CaptchaBreaker/
├── app/                          # Main application
│   ├── __init__.py              # Package metadata
│   └── main.py                  # FastAPI server + OCR logic
├── data/                         # Data directory
│   └── samples/                 # Test captcha images (31 samples)
├── docs/                         # Documentation
│   ├── ARCHITECTURE.md          # Technical architecture details
│   ├── OPTIMIZATION.md          # Optimization roadmap & suggestions
│   └── API.md                   # API reference documentation
├── scripts/                      # Utility scripts
│   ├── test_accuracy.py         # Batch accuracy testing
│   └── test_single.py           # Single image testing
├── tests/                        # Test suite (future)
├── assets/                       # Static assets (screenshots, etc.)
├── run.py                        # Application entry point
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ocr/upload` | POST | Upload image file for recognition |
| `/ocr/base64` | POST | Send Base64-encoded image |
| `/ocr/url` | POST | Provide image URL for recognition |
| `/health` | GET | Server health check |
| `/` | GET | API info and version |

### Response Format

```json
{
    "success": true,
    "image_id": "a1b2c3d4e5f6...",
    "texts": [
        {
            "text": "A7k9",
            "confidence": 0.9956,
            "bounding_box": {
                "x_min": 75.0,
                "y_min": 17.0,
                "x_max": 200.0,
                "y_max": 69.0
            }
        }
    ],
    "full_text": "A7k9",
    "language": "en",
    "message": "Successfully recognized 1 text(s)",
    "preprocessing_applied": [
        "Upscale from 100x40 to 200x80",
        "Light contrast enhancement"
    ]
}
```

## Performance

### Current Accuracy (v5.0)

| Metric | Value | Notes |
|--------|-------|-------|
| **True Accuracy** | ~60% | Exact match against filename |
| **Avg Confidence** | ~0.85 | OCR model self-assessment |
| **Avg Latency** | ~3.5s | CPU mode, first inference |
| **Sample Size** | 31 images | Mixed captcha types |

### Why Not Higher?

Captchas are specifically designed to **defeat automated recognition**. Common challenges include:

- Character粘连 (characters touching or overlapping)
- Interference lines crossing through characters
- Unconventional fonts not in training data
- Small image sizes (~100x40px) with limited pixel information

### Path to 90%+ Accuracy

See [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) for detailed improvement strategies:

1. **Model Fine-tuning** (90-95%) - Train on 1000+ same-type captchas
2. **Multi-Strategy Ensemble** (75-85%) - Vote across multiple approaches
3. **Targeted Preprocessing** (70-80%) - Custom preprocessing per captcha type

## Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| OCR Engine | PaddleOCR (PP-OCRv5) | 3.4.1 |
| Deep Learning | PaddlePaddle | 3.0.0 |
| Web Framework | FastAPI | 0.115.0 |
| Image Processing | Pillow + OpenCV | 10.4.0 + 4.10.0 |
| Numerical Computing | NumPy | 1.26.4 |

## Usage Examples

### Python

```python
import requests

# Upload method
with open("captcha.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/ocr/upload",
        files={"file": f},
        data={"language": "en"}
    )
    print(response.json()["full_text"])

# Base64 method
import base64
with open("captcha.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
    response = requests.post(
        "http://localhost:8000/ocr/base64",
        data={"image_data": b64, "language": "en"}
    )
    print(response.json()["full_text"])
```

### cURL

```bash
# File upload
curl -X POST "http://localhost:8000/ocr/upload" \
  -F "file=@captcha.png" \
  -F "language=en"

# Base64
curl -X POST "http://localhost:8000/ocr/base64" \
  -d "image_data=data:image/png;base64,iVBORw..." \
  -d "language=en"

# URL
curl -X POST "http://localhost:8000/ocr/url" \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/captcha.png", "language": "en"}'
```

## Documentation

- [Technical Architecture](docs/ARCHITECTURE.md) - System design and core technologies
- [Optimization Roadmap](docs/OPTIMIZATION.md) - How to improve accuracy to 90%+
- [API Reference](docs/API.md) - Detailed API documentation

## Requirements

- Python 3.10+
- Windows / Linux / macOS
- Memory >= 4GB
- Disk space >= 2GB (model files ~500MB)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

[zhuzhiqianggg](https://github.com/zhuzhiqianggg)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
