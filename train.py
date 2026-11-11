"""Huấn luyện YOLO26 Detection với dataset phương tiện Việt Nam.
Chỉ chạy khi đã có ảnh và file bbox gán nhãn. Không có dataset = KHÔNG thể train.
"""
import argparse
import shutil
from pathlib import Path

import yaml

from check_dataset import validate
from vehicle_ai.labels import VN_LABELS


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Train custom VehicleAI VN')
    parser.add_argument('--base', default='yolo26s.pt', help='Pretrained YOLO detection model (n/s/m/l)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default='cpu', help='cpu hoặc 0 cho NVIDIA CUDA')
    parser.add_argument('--workers', type=int, default=0, help='Windows: 0 để tránh multiprocessing lỗi')
    args = parser.parse_args()
    images, counts, errors = validate()
    if errors or images < 10:
        raise SystemExit('Dataset chưa đủ/không hợp lệ. Gán nhãn ảnh rồi chạy python check_dataset.py')
    missing_train = [name for i, name in enumerate(VN_LABELS) if counts['train', i] == 0]
    if missing_train:
        print('[CẢNH BÁO] Các lớp chưa có mẫu train nên model KHÔNG thể học:')
        print(', '.join(missing_train))
    # Đường dẫn tuyệt đối để không phụ thuộc Ultralytics datasets_dir trên Windows.
    cfg = {
        'path': str((ROOT / 'dataset').resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': list(VN_LABELS),
    }
    generated = ROOT / 'data' / 'vehicles_vn_local.yaml'
    generated.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    from ultralytics import YOLO
    model = YOLO(args.base)
    model.train(data=str(generated), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, workers=args.workers,
                project=str(ROOT / 'runs'), name='vehicles_vn', exist_ok=True,
                plots=True, patience=25)
    best = ROOT / 'runs/vehicles_vn/weights/best.pt'
    if best.exists():
        destination = ROOT / 'weights/vehicles_vn_best.pt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, destination)
        print('[OK] Đã lưu:', destination)
        print('[LƯU Ý] Phải kiểm tra mAP và confusion matrix trên tập test riêng.')
    else:
        raise SystemExit('Chưa thấy best.pt, xem log huấn luyện trong runs/vehicles_vn')


if __name__ == '__main__':
    main()
