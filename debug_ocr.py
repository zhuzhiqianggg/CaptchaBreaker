import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import PaddleOCR
import json

ocr = PaddleOCR(lang='en')
result = ocr.ocr("test_images/test_captcha.png")

print("Result type:", type(result))
print("Result:", result)
print("\nResult[0] type:", type(result[0]))
print("Result[0]:", result[0])

if result[0]:
    print("\nFirst item type:", type(result[0][0]))
    print("First item:", result[0][0])
    if len(result[0][0]) > 0:
        print("\nFirst item[0]:", result[0][0][0])
    if len(result[0][0]) > 1:
        print("First item[1]:", result[0][0][1])
