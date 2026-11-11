"""Bộ vẽ bbox YOLO cơ bản để tự gán nhãn trong VS Code.

Chạy: python label_dataset.py --split train (hoặc val)
C: đổi mã lớp bằng terminal | giữ chuột trái và kéo khung | Z: undo
S: lưu và sang ảnh sau | N: bỏ qua ảnh | Q: thoát
Gán nhãn theo lớp trong data/vehicles_vn.yaml, KHÔNG suy đoán từ màu/logo.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vehicle_ai.labels import VN_LABELS
from vehicle_ai.runner import imread_unicode

ROOT = Path(__file__).resolve().parent


def show_classes():
    print('\nDANH MỤC MÃ LỚP:')
    for i, (slug, vn) in enumerate(VN_LABELS.items()):
        print(f'{i:>2}: {vn}  ({slug})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train', 'val'], default='train')
    args = parser.parse_args()
    folder = ROOT / 'dataset/images' / args.split
    out = ROOT / 'dataset/labels' / args.split
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}] if folder.exists() else []
    if not files:
        raise SystemExit(f'Chưa có ảnh trong {folder}')
    show_classes()
    try:
        selected = int(input('Nhập mã lớp hiện tại (0..47): '))
    except ValueError:
        raise SystemExit('Phải nhập số nguyên.')
    if selected not in range(len(VN_LABELS)):
        raise SystemExit('Mã lớp không hợp lệ')
    state = {'boxes': [], 'drawing': False, 'start': None, 'end': None, 'selected': selected}
    for path in files:
        image = imread_unicode(path)
        H, W = image.shape[:2]
        label_path = out/(path.stem+'.txt')
        state.update(boxes=[], drawing=False, start=None, end=None)
        if label_path.exists():
            for row in label_path.read_text(encoding='utf8').splitlines():
                cols = row.split()
                if len(cols) != 5:
                    continue
                try:
                    c, xc, yc, w, h = [float(x) for x in cols]
                    state['boxes'].append((int(c), int((xc-w/2)*W), int((yc-h/2)*H),
                                           int((xc+w/2)*W), int((yc+h/2)*H)))
                except ValueError:
                    pass
        window = 'GAN NHAN | C:class | S:save | Z:undo | N:next | Q:quit'
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        def callback(event, x, y, flags, param):
            x, y = max(0, min(W-1, x)), max(0, min(H-1, y))
            if event == cv2.EVENT_LBUTTONDOWN:
                state['drawing'] = True
                state['start'] = (x, y)
                state['end'] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and state['drawing']:
                state['end'] = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and state['drawing']:
                state['drawing'] = False
                xx, yy = state['start']
                x1, x2, y1, y2 = min(xx, x), max(xx, x), min(yy, y), max(yy, y)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    state['boxes'].append((state['selected'], x1, y1, x2, y2))
        cv2.setMouseCallback(window, callback)
        print(f'\nẢnh: {path.name} - nhãn: {len(state["boxes"])} - lớp: {state["selected"]} {VN_LABELS[list(VN_LABELS)[state["selected"]]]}')
        while True:
            display = image.copy()
            for c, x1, y1, x2, y2 in state['boxes']:
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 160), 2)
                cv2.putText(display, str(c), (x1, max(17, y1-4)), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, (0, 255, 160), 2)
            if state['drawing'] and state['start'] and state['end']:
                cv2.rectangle(display, state['start'], state['end'], (30, 175, 255), 2)
            cv2.putText(display, f'Class:{state["selected"]}  Boxes:{len(state["boxes"])}',
                        (6, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 3)
            cv2.putText(display, f'Class:{state["selected"]}  Boxes:{len(state["boxes"])}',
                        (6, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 0), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('c'):
                try:
                    new = int(input('Mã lớp mới: '))
                    if new in range(len(VN_LABELS)):
                        state['selected'] = new
                except ValueError:
                    print('Mã lớp phải là số nguyên')
            if key == ord('z') and state['boxes']:
                state['boxes'].pop()
            if key == ord('s'):
                rows = []
                for c, x1, y1, x2, y2 in state['boxes']:
                    rows.append(f'{c} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}')
                label_path.write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf8')
                print('[OK] Đã lưu:', label_path)
                break
            if key in (ord('n'), ord('q'), 27):
                break
        cv2.destroyWindow(window)
        if key in (ord('q'), 27):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
