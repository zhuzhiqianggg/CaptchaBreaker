import requests

with open("code_images/2zrw.png", "rb") as f:
    r = requests.post("http://localhost:8000/ocr/upload", files={"file": f}, data={"language": "en"})
    print(r.status_code)
    print(r.text[:500])
