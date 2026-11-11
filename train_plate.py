"""Huấn luyện detector biển số YOLO tách biệt với detector 48 lớp phương tiện.

Ví dụ:
  python train_plate.py --data data/plate.yaml --epochs 100 --device 0
Kết quả tốt nhất được chép thành weights/license_plate_detector.pt.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Vietnamese license-plate detector")
    parser.add_argument("--data", default="dataset/license_plate/data.yaml", help="YOLO data.yaml của bộ biển số")
    parser.add_argument("--base", default="yolo26n.pt", help="Model YOLO detection khởi tạo")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu hoặc 0 cho NVIDIA CUDA")
    parser.add_argument("--workers", type=int, default=0, help="Windows: 0")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.is_absolute():
        data = (ROOT / data).resolve()
    if not data.is_file():
        raise SystemExit(f"Không tìm thấy {data}. Hãy tạo data/plate.yaml theo dataset YOLO đã tải.")

    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                project=str(ROOT / "runs"), name="license_plate", exist_ok=True,
                patience=25, plots=True)

    best = ROOT / "runs/license_plate/weights/best.pt"
    if not best.is_file():
        raise SystemExit("Không thấy best.pt. Kiểm tra log trong runs/license_plate.")
    target = ROOT / "weights/license_plate_detector.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print(f"[OK] Đã lưu detector biển số: {target}")
    print("Lưu ý: đánh giá precision/recall/mAP trên test độc lập trước khi sử dụng.")


if __name__ == "__main__":
    main()
