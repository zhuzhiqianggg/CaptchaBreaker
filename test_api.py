import requests
from PIL import Image, ImageDraw, ImageFont
import json
import base64
import os
import time

BASE_URL = "http://localhost:8000"

def create_test_images():
    os.makedirs("test_images", exist_ok=True)

    img = Image.new('RGB', (200, 80), color='white')
    draw = ImageDraw.Draw(img)
    
    text = "A7k9"
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    
    draw.text((20, 10), text, font=font, fill='black')
    for i in range(5):
        draw.line([(0, i*20), (200, i*20)], fill='gray')
    
    img.save("test_images/test_captcha.png")
    print(f"Created: test_images/test_captcha.png")

    img2 = Image.new('RGB', (300, 100), color='white')
    draw2 = ImageDraw.Draw(img2)
    draw2.text((10, 20), "Hello 世界", font=font, fill='black')
    img2.save("test_images/test_chinese.png")
    print(f"Created: test_images/test_chinese.png")

def test_upload_endpoint():
    print("\n=== Test 1: Upload Image File ===")
    url = f"{BASE_URL}/ocr/upload"
    
    with open("test_images/test_captcha.png", "rb") as f:
        files = {"file": ("test_captcha.png", f, "image/png")}
        data = {"language": "en"}
        start_time = time.time()
        response = requests.post(url, files=files, data=data)
        elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"Time: {elapsed:.2f}s")
    assert response.status_code == 200
    assert result["success"] == True
    print("✓ Test passed!")

def test_base64_endpoint():
    print("\n=== Test 2: Base64 Image Data ===")
    url = f"{BASE_URL}/ocr/base64"
    
    with open("test_images/test_chinese.png", "rb") as f:
        image_bytes = f.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    data = {
        "image_data": f"data:image/png;base64,{image_base64}",
        "language": "general"
    }
    start_time = time.time()
    response = requests.post(url, data=data)
    elapsed = time.time() - start_time
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
    print(f"Time: {elapsed:.2f}s")
    assert response.status_code == 200
    assert result["success"] == True
    print("✓ Test passed!")

def test_health_endpoint():
    print("\n=== Test 3: Health Check ===")
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Test passed!")

def test_root_endpoint():
    print("\n=== Test 4: Root Endpoint ===")
    url = f"{BASE_URL}/"
    response = requests.get(url)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✓ Test passed!")

if __name__ == "__main__":
    print("Creating test images...")
    create_test_images()
    
    test_upload_endpoint()
    test_base64_endpoint()
    test_health_endpoint()
    test_root_endpoint()
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
