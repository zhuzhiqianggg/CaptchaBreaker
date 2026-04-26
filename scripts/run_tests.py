#!/usr/bin/env python3
"""
统一的OCR测试入口脚本
支持本地模式和API模式测试
"""

import sys
import os
import time
import json
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "samples"
BASE_URL = "http://localhost:8000"

def test_api_mode():
    """通过API接口测试OCR服务"""
    imgs = sorted(SAMPLES_DIR.glob("*.png"))
    if not imgs:
        print("错误: 未找到测试图片")
        return

    total = len(imgs)
    print(f"\n{'='*70}")
    print("OCR API 测试")
    print(f"测试样本数: {total}")
    print(f"{'='*70}\n")

    ok = 0
    fail = 0
    results = []

    for idx, img in enumerate(imgs, 1):
        exp = img.stem.lower()
        try:
            with open(img, "rb") as f:
                r = requests.post(
                    f"{BASE_URL}/ocr/upload",
                    files={"file": f},
                    data={"language": "en"},
                    timeout=30
                )
            result = r.json()
            got = result.get("full_text", "").lower().replace(" ", "")
            match = (got == exp)
            if match:
                ok += 1
            else:
                fail += 1
            conf = result["texts"][0]["confidence"] if result.get("texts") else 0
            tag = "OK" if match else "FAIL"
            print(f"  [{idx:2d}/{total}] {tag:4s} {img.name:15s} | 期望: {exp:8s} 识别: {got:10s} 置信度: {conf:.3f}")
            sys.stdout.flush()

            results.append({
                "file": img.name,
                "expected": exp,
                "got": got,
                "match": match,
                "confidence": conf
            })
        except Exception as e:
            fail += 1
            print(f"  [{idx:2d}/{total}] ERR  {img.name:15s} | {str(e)[:50]}")
            sys.stdout.flush()

    print(f"\n{'='*70}")
    print(f"测试结果: {ok}/{total} 正确")
    print(f"准确率: {ok/total*100:.1f}%")
    print(f"{'='*70}\n")

    if fail > 0:
        print("失败案例:")
        for r in results:
            if not r["match"]:
                print(f"  {r['file']:15s} 期望: {r['expected']:8s} 识别: {r['got']}")
        print()

    return ok, total


def test_local_mode():
    """直接加载OCR模型进行测试"""
    try:
        from app.main import (
            preprocess_image_v1,
            preprocess_image_v2,
            preprocess_image_v3,
            preprocess_image_v4,
            vote_results,
            smart_correct,
            parse_ocr_result
        )
        from paddleocr import PaddleOCR
        from PIL import Image
    except ImportError as e:
        print(f"错误: 无法导入必要模块 - {e}")
        print("请确保已安装 PaddleOCR: pip install paddleocr")
        return None, None

    imgs = sorted(SAMPLES_DIR.glob("*.png"))
    if not imgs:
        print("错误: 未找到测试图片")
        return

    print(f"\n{'='*70}")
    print("CaptchaBreaker OCR 本地模式测试")
    print(f"测试样本数: {len(imgs)}")
    print(f"{'='*70}\n")

    print("正在加载 PaddleOCR 模型...")
    ocr = PaddleOCR(lang='en', show_log=False)
    print("模型加载完成!\n")

    total = 0
    correct = 0
    corrected_count = 0
    results = []

    for img_path in imgs:
        image = Image.open(img_path)
        expected = img_path.stem.lower().strip()
        total += 1

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
        corrected = smart_correct(voted, expected_length=4)

        if corrected == expected:
            correct += 1
            status = "[OK]"
        elif best_text.lower() == expected:
            correct += 1
            status = "[OK]"
        else:
            status = "[FAIL]"

        if corrected != voted and voted.lower() != expected and corrected.lower() == expected:
            corrected_count += 1
            status = "[OK*]"

        print(f"{img_path.name:15s} 期望: {expected:8s} 识别: {voted:12s} 纠错: {corrected:12s} {status}")
        results.append({
            'file': img_path.name,
            'expected': expected,
            'voted': voted,
            'corrected': corrected,
            'match': (corrected == expected or best_text.lower() == expected)
        })

    print(f"\n{'='*70}")
    print(f"测试结果: {correct}/{total} 正确")
    print(f"准确率: {correct/total*100:.1f}%")
    if corrected_count > 0:
        print(f"智能纠错成功: {corrected_count} 个")
    print(f"{'='*70}\n")

    if correct < total:
        print("失败案例:")
        for r in results:
            if not r['match']:
                print(f"  {r['file']:15s} 期望: {r['expected']:8s} 识别: {r['voted']:12s} 纠错: {r['corrected']}")
        print()

    return correct, total


def test_single_image(image_path):
    """测试单张图片"""
    img = Path(image_path)
    if not img.exists():
        print(f"文件不存在: {img}")
        return

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        is_api_running = response.status_code == 200
    except:
        is_api_running = False

    if is_api_running:
        exp = img.stem.lower()
        print(f"测试: {img.name}")
        print(f"期望: {exp}\n")

        with open(img, "rb") as f:
            r = requests.post(f"{BASE_URL}/ocr/upload",
                            files={"file": f},
                            data={"language": "en"},
                            timeout=30)

        if r.status_code == 200:
            result = r.json()
            got = result.get("full_text", "").lower().replace(" ", "")
            match = "匹配" if got == exp else "不匹配"

            print(f"识别结果: {got}")
            print(f"状态: {match}")
            print(f"置信度: {result['texts'][0]['confidence'] if result.get('texts') else 0:.3f}")
            print(f"预处理: {result.get('preprocessing_applied', [])}")
        else:
            print(f"错误: HTTP {r.status_code}")
            print(r.text)
    else:
        print("API 服务未运行，切换到本地模式...")
        try:
            from app.main import (
                preprocess_image_v1, vote_results, smart_correct, parse_ocr_result
            )
            from paddleocr import PaddleOCR
            from PIL import Image

            print("加载模型...")
            ocr = PaddleOCR(lang='en', show_log=False)

            image = Image.open(img)
            processed, steps = preprocess_image_v1(image)

            temp_path = Path(__file__).parent.parent / "temp" / "test_single.png"
            temp_path.parent.mkdir(exist_ok=True)
            processed.save(temp_path)

            result = ocr.ocr(str(temp_path))
            texts, full_text = parse_ocr_result(result)

            print(f"\n识别结果: {full_text}")
            if texts:
                print(f"置信度: {texts[0].confidence:.3f}")
        except ImportError:
            print("错误: 无法导入必要模块，请安装 PaddleOCR")


def test_health():
    """测试服务健康状态"""
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code == 200:
            print("✅ API 服务正常")
            print(f"响应: {r.json()}")
            return True
        else:
            print(f"❌ API 异常: HTTP {r.status_code}")
            return False
    except requests.ConnectionError:
        print("❌ API 服务未运行")
        print(f"提示: 请先启动服务 python run.py 或 docker-compose up -d")
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "api":
            test_api_mode()
        elif command == "local":
            test_local_mode()
        elif command == "single":
            if len(sys.argv) > 2:
                test_single_image(sys.argv[2])
            else:
                print("用法: python run_tests.py single <image_path>")
        elif command == "health":
            test_health()
        else:
            print(f"未知命令: {command}")
            print("可用命令: api, local, single, health")
    else:
        print("="*70)
        print("CaptchaBreaker OCR 测试工具")
        print("="*70)
        print("\n请选择测试模式:\n")
        print("  1. API 模式 - 测试已启动的 OCR 服务")
        print("  2. 本地模式 - 直接加载模型进行测试")
        print("  3. 单图测试 - 测试单个图片")
        print("  4. 健康检查 - 检查服务状态")
        print("\n用法:")
        print("  python run_tests.py api          # API 批量测试")
        print("  python run_tests.py local        # 本地模式测试")
        print("  python run_tests.py single <file> # 单图测试")
        print("  python run_tests.py health       # 健康检查")
        print("="*70)
