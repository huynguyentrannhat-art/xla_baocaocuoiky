"""Chạy: python main.py --source 0 | --source input/anh.jpg | --source input/video.mp4"""
import argparse
from pathlib import Path

from vehicle_ai.engine import VehicleEngine
from vehicle_ai.runner import IMAGE_EXTS, VIDEO_EXTS, run_image, run_video


def cli():
    ap = argparse.ArgumentParser(description='VehicleAI VN - nhận diện xe và màu sắc')
    ap.add_argument('--source', default='0', help='Đường dẫn ảnh/video hoặc số camera 0/1')
    ap.add_argument('--weights', default='weights/vehicles_vn_coarse_cls.pt',help='Model phân loại xe 4 nhóm')
    ap.add_argument('--conf', type=float, default=.38)
    ap.add_argument('--imgsz', type=int, default=768)
    ap.add_argument('--device', default='', help='cpu hoặc 0 (GPU NVIDIA)')
    ap.add_argument('--no-seg', action='store_true', help='Bỏ model phân vùng màu ở chế độ nâng cao')
    ap.add_argument('--show', action='store_true', help='Hiện cửa sổ video (camera tự bật)')
    ap.add_argument('--no-show', action='store_true', help='Không hiện cửa sổ camera')
    ap.add_argument('--max-frames', type=int, default=None, help='Giới hạn frame kiểm thử')
    ap.add_argument('--output', default='output')
    args = ap.parse_args()
    weights_path = Path(args.weights) if args.weights else Path('weights/vehicles_vn_coarse_cls.pt')
    weights = str(weights_path) if weights_path.is_file() else None
    if args.weights and weights is None:
        ap.error(f'Không tìm thấy mô hình riêng: {args.weights}')
    print('[INFO] Chế độ:', 'PHÂN LOẠI CHI TIẾT (custom)' if weights else 'COCO CƠ BẢN')
    print('[INFO] Lần đầu có thể tự tải yolo26n-seg.pt qua Internet.')
    engine = VehicleEngine(weights, conf=args.conf, imgsz=args.imgsz,
                           device=args.device or None, use_seg=not args.no_seg)
    out = Path(args.output)
    source = args.source
    path = Path(source)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        saved, csv_path, json_path, vehicles, _ = run_image(engine, path, out)
        print(f'[OK] Nhận diện {len(vehicles)} phương tiện:')
        for v in vehicles:
            print(f'  - {v.label} | {v.color} | xe {v.confidence:.0%} | màu {v.color_confidence:.0%}' + (f' | biển số {v.plate_text}' if v.plate_text else ''))
    elif source.isdigit() and not path.is_file():
        show = not args.no_show
        saved, csv_path, json_path, _ = run_video(engine, int(source), out,
                                                   show=show, max_frames=args.max_frames)
    elif path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        saved, csv_path, json_path, _ = run_video(engine, str(path), out,
                    show=args.show and not args.no_show, max_frames=args.max_frames)
    else:
        ap.error(f'Nguồn không hợp lệ hoặc file không tồn tại: {source}')
    print('[OK] Kết quả:', saved)
    print('[OK] CSV:', csv_path)
    print('[OK] JSON:', json_path)


if __name__ == '__main__':
    cli()
