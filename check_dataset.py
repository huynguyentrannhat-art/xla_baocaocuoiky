"""Kiểm tra dataset YOLO: image, label, class index và số mẫu từng lớp."""
from collections import Counter
from pathlib import Path

from vehicle_ai.labels import VN_LABELS

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'dataset'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def validate(quiet=False):
    n_classes = len(VN_LABELS)
    counts = Counter()
    errors = []
    n_images = 0
    for split in ['train', 'val']:
        image_folder = DATA / 'images' / split
        label_folder = DATA / 'labels' / split
        images = [p for p in image_folder.iterdir() if p.suffix.lower() in IMAGE_EXTS] if image_folder.exists() else []
        if not images:
            errors.append(f'Không có ảnh trong {image_folder}')
        for image in images:
            n_images += 1
            label = label_folder / (image.stem + '.txt')
            if not label.exists():
                errors.append(f'Thiếu nhãn: {label}')
                continue
            for index, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.strip().split()
                if len(fields) != 5:
                    errors.append(f'{label}:{index}: cần 5 cột class x_center y_center width height')
                    continue
                try:
                    class_id = int(fields[0])
                    values = [float(x) for x in fields[1:]]
                except ValueError:
                    errors.append(f'{label}:{index}: giá trị không hợp lệ')
                    continue
                if class_id < 0 or class_id >= n_classes:
                    errors.append(f'{label}:{index}: class_id phải thuộc [0,{n_classes-1}]')
                else:
                    counts[(split, class_id)] += 1
                if any(not (0 <= x <= 1) for x in values) or values[2] <= 0 or values[3] <= 0:
                    errors.append(f'{label}:{index}: tọa độ YOLO phải trong 0..1 và w,h > 0')
    if not quiet:
        print('Ảnh:', n_images, '| Số nhãn:', sum(counts.values()))
        print('\nMẫu vật thể theo từng lớp (train / val):')
        for i, (key, vn) in enumerate(VN_LABELS.items()):
            print(f'{i:>2} {key:<28} {counts["train", i]:>5} / {counts["val", i]:>5} | {vn}')
        if errors:
            print('\nLỖI:')
            for x in errors[:35]:
                print(' -', x)
            if len(errors) > 35:
                print(f' ... còn {len(errors)-35} lỗi')
        else:
            print('\n[OK] Cấu trúc cơ bản hợp lệ. Đây KHÔNG phải kiểm định chất lượng nhãn.')
    return n_images, counts, errors


if __name__ == '__main__':
    _, _, errors = validate()
    raise SystemExit(1 if errors else 0)
