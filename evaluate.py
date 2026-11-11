"""Đánh giá model bằng tập val/test, không tự tạo số liệu độ chính xác."""
import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Đánh giá mAP50, mAP50-95')
    parser.add_argument('--weights', default='weights/vehicles_vn_best.pt')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    path = ROOT / args.weights
    if not path.is_file():
        raise SystemExit(f'Chưa có: {path}. Chạy train.py trước.')
    split_dir = ROOT / 'dataset/images' / args.split
    if not split_dir.exists() or not any(x.is_file() and x.suffix.lower() in {'.jpg','.png','.jpeg'} for x in split_dir.iterdir()):
        raise SystemExit(f'Chưa có ảnh cho tập {args.split}: {split_dir}')
    from vehicle_ai.labels import VN_LABELS
    cfg = {'path': str((ROOT / 'dataset').resolve()), 'train': 'images/train',
           'val': 'images/val', 'test': 'images/test', 'names': list(VN_LABELS)}
    cfgpath = ROOT/'data/vehicles_vn_local.yaml'
    cfgpath.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    from ultralytics import YOLO
    model = YOLO(str(path))
    metrics = model.val(data=str(cfgpath), split=args.split, device=args.device,
                        project=str(ROOT/'runs'), name='danhgia_'+args.split,
                        exist_ok=True, plots=True)
    print('mAP50:', round(float(metrics.box.map50), 4))
    print('mAP50-95:', round(float(metrics.box.map), 4))
    print('Đồ thị đánh giá:', ROOT/'runs'/('danhgia_'+args.split))


if __name__ == '__main__':
    main()
