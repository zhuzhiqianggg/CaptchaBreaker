import requests
from PIL import Image, ImageDraw, ImageFont
import json
import os

print("Creating test captcha image...")

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

os.makedirs("test_images", exist_ok=True)
img.save("test_images/test_captcha.png")
print(f"Test image saved: test_images/test_captcha.png")

print("\nTesting OCR upload endpoint...")
url = "http://localhost:8000/ocr/upload"

with open("test_images/test_captcha.png", "rb") as f:
    files = {"file": ("test_captcha.png", f, "image/png")}
    data = {"language": "en"}
    response = requests.post(url, files=files, data=data)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

if response.status_code == 200:
    result = response.json()
    print(f"\nRecognized text: '{result['full_text']}'")
    print(f"Success: {result['success']}")
else:
    print(f"Error: {response.text}")
