#!/usr/bin/env python3
"""
数据增强脚本 - 将现有验证码图片扩充到 1000+ 张
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

# 原始数据目录
DATA_DIR = Path(__file__).parent.parent / "data" / "samples"
# 增强数据输出目录
AUGMENTED_DIR = Path(__file__).parent.parent / "data" / "finetune" / "train"

AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)


def add_noise(image, intensity=0.02):
    """添加随机噪声"""
    img_array = np.array(image)
    noise = np.random.normal(0, 255 * intensity, img_array.shape).astype(np.int16)
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def add_gaussian_noise(image, mean=0, std=10):
    """添加高斯噪声"""
    img_array = np.array(image)
    noise = np.random.normal(mean, std, img_array.shape).astype(np.int16)
    noisy_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_array)


def random_rotate(image, max_angle=10):
    """随机旋转"""
    angle = random.uniform(-max_angle, max_angle)
    return image.rotate(angle, resample=Image.BICUBIC, expand=False)


def random_crop(image, crop_ratio=0.9):
    """随机裁剪"""
    w, h = image.size
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return image.crop((left, top, left + new_w, top + new_h)).resize((w, h))


def random_contrast(image, factor_range=(0.8, 1.5)):
    """随机对比度"""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def random_brightness(image, factor_range=(0.7, 1.3)):
    """随机亮度"""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def random_sharpness(image, factor_range=(0.5, 2.0)):
    """随机锐化"""
    factor = random.uniform(*factor_range)
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(factor)


def apply_blur(image, radius=1):
    """应用模糊"""
    if radius == 1:
        return image.filter(ImageFilter.GaussianBlur(radius=1))
    return image.filter(ImageFilter.BoxBlur(radius))


def add_random_lines(image, num_lines=3):
    """添加随机干扰线"""
    img_array = np.array(image)
    h, w, _ = img_array.shape
    
    for _ in range(num_lines):
        y = random.randint(0, h - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_array[y, :, :] = color
    
    return Image.fromarray(img_array)


def augment_single_image(image_path, output_dir, num_augmentations=30):
    """对单张图片进行数据增强"""
    filename = image_path.stem
    original = Image.open(image_path).convert("RGB")
    
    augmentations = [
        add_noise,
        add_gaussian_noise,
        lambda img: random_rotate(img, max_angle=8),
        lambda img: random_crop(img, crop_ratio=0.95),
        lambda img: random_contrast(img, factor_range=(0.7, 1.8)),
        lambda img: random_brightness(img, factor_range=(0.6, 1.4)),
        lambda img: random_sharpness(img, factor_range=(0.3, 2.5)),
        lambda img: apply_blur(img, radius=1),
        lambda img: add_random_lines(img, num_lines=random.randint(1, 5)),
    ]
    
    # 保存原始图片
    original.save(output_dir / f"{filename}_0.png")
    
    # 生成增强图片
    for i in range(1, num_augmentations):
        img = original.copy()
        
        num_transforms = random.randint(1, 3)
        transforms = random.sample(augmentations, num_transforms)
        
        for transform in transforms:
            if random.random() < 0.7:
                img = transform(img)
        
        img.save(output_dir / f"{filename}_{i}.png")
    
    return num_augmentations


def main():
    print("=" * 60)
    print("数据增强 - 扩充训练数据集")
    print("=" * 60)
    
    images = sorted(DATA_DIR.glob("*.png"))
    if not images:
        print("未找到训练图片！")
        return
    
    print(f"\n原始图片数量: {len(images)}")
    print(f"目标数量: {len(images) * 32}")
    
    total_augmented = 0
    for img_path in images:
        count = augment_single_image(img_path, AUGMENTED_DIR, num_augmentations=32)
        total_augmented += count
        print(f"  增强 {img_path.name} -> {count} 张")
    
    print(f"\n总计生成: {total_augmented} 张图片")
    print(f"输出目录: {AUGMENTED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
