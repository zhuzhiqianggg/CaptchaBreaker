#!/usr/bin/env python3
"""
PaddleOCR Fine-tuning 训练脚本
使用增强后的验证码数据微调识别模型
"""

import os
import sys
from pathlib import Path
import json

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "finetune"
OUTPUT_DIR = PROJECT_DIR / "output" / "finetune"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_train_config():
    """创建训练配置文件"""
    
    config = {
        "Global": {
            "use_gpu": False,
            "epoch_num": 50,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": str(OUTPUT_DIR),
            "save_epoch_step": 5,
            "eval_batch_step": [0, 50],
            "cal_metric_during_train": True,
            "pretrained_model": None,
            "output_dir": str(OUTPUT_DIR),
            "checkpoints": None,
            "save_inference_dir": str(OUTPUT_DIR / "inference"),
            "use_visualdl": False,
            "infer_img": None,
            "save_res_path": str(OUTPUT_DIR / "rec_train_predict.txt"),
        },
        "Architecture": {
            "function": "paddleocr.ppocr.modeling.architectures.BaseModel",
            "Transform": None,
            "Backbone": {
                "function": "paddleocr.ppocr.modeling.backbones.MobileNetV3",
                "scale": 0.5,
                "model_name": "small",
                "small_stride": [1, 2, 2, 2],
            },
            "Neck": {
                "function": "paddleocr.ppocr.modeling.necks.SequenceEncoder",
                "encoder_type": "rnn",
                "hidden_size": 48,
            },
            "Head": {
                "function": "paddleocr.ppocr.modeling.heads.CTCHead",
                "fc_decay": 0.00001,
            },
        },
        "Loss": {
            "function": "paddleocr.ppocr.modeling.losses.CTCLoss",
        },
        "Optimizer": {
            "function": "paddleocr.ppocr.optimizer.Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "function": "paddleocr.ppocr.optimizer.lr.Cosine",
                "learning_rate": 0.0005,
                "epoch_num": 50,
            },
            "regularizer": {
                "function": "L2",
                "factor": 0.00001,
            },
        },
    }
    
    config_file = OUTPUT_DIR / "train_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ 训练配置已生成: {config_file}")
    return config_file


def main():
    print("=" * 70)
    print("PaddleOCR Fine-tuning 训练")
    print("=" * 70)
    
    print("\n[1/2] 生成训练配置...")
    config_file = create_train_config()
    
    print("\n[2/2] 开始训练...")
    print(f"  训练数据: {DATA_DIR / 'train'}")
    print(f"  标注文件: {DATA_DIR / 'rec_gt_train_split.txt'}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  配置参数:")
    print(f"    - epoch_num: 50")
    print(f"    - learning_rate: 0.0005")
    print(f"    - batch_size: 256")
    print(f"    - use_gpu: False")
    
    print(f"\n{'='*70}")
    print("训练准备完成！")
    print(f"{'='*70}\n")
    print("注意: PaddleOCR 3.x 使用新的训练 API")
    print("推荐使用 PaddleX 进行可视化训练")
    print("\n或者使用命令:")
    print("  python tools/train.py -c configs/rec/rec_mv3_en_PP-OCRv5_train.yml")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
