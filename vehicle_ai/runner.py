"""Ảnh/video/camera, lưu file kết quả và bảng phân loại."""
from pathlib import Path

import cv2
import numpy as np

from .drawing import annotate
from .reports import rows_for, write_reports

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}


def imread_unicode(path):
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Không đọc được ảnh: {path}')
    return img


def imwrite_unicode(path, img):
    suffix = Path(path).suffix or '.png'
    ok, data = cv2.imencode(suffix, img)
    if not ok:
        raise OSError(f'Không ghi được ảnh: {path}')
    data.tofile(str(path))


def run_image(engine, source: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    image = imread_unicode(source)
    objects = engine.process(image)
    overlay = annotate(image, objects, total=len(objects))
    dst = out_dir / (source.stem + '_ketqua.png')
    imwrite_unicode(dst, overlay)
    prefix = out_dir / (source.stem + '_thongke')
    summary = write_reports(rows_for(objects), prefix, {
        'nguon': str(source), 'che_do': 'nang_cao' if engine.is_custom else 'co_ban',
        'so_xe_trong_anh': len(objects),
    })
    return dst, prefix.with_suffix('.csv'), prefix.with_suffix('.json'), objects, summary


def run_video(engine, source: str | int, out_dir: Path, show=False,
              max_frames: int | None = None, prefix: str | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f'Không mở được nguồn {source}. Camera thường là 0, thử --source 1.')
    name = prefix or ('camera' if isinstance(source, int) else Path(source).stem)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps < 1 or fps > 240:
        fps = 25.
    writer = None
    video_path = out_dir / (name + '_ketqua.mp4')
    rows, frames, peak = [], 0, 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Không tạo được video kết quả MP4. Kiểm tra codec OpenCV.')
            objects = engine.process(frame, tracking=True)
            overlay = annotate(frame, objects, total=len(objects))
            writer.write(overlay)
            frames += 1
            peak = max(peak, len(objects))
            rows.extend(rows_for(objects, frame=frames, time_seconds=(frames-1)/fps))
            if show:
                cv2.imshow('Nhan dien phuong tien - nhan Q de dung', overlay)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            if max_frames is not None and frames >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()
    report_prefix = out_dir / (name + '_thongke')
    summary = write_reports(rows, report_prefix, {
        'nguon': str(source), 'so_khung_hinh': frames,
        'fps_nguon': fps, 'so_xe_cao_nhat_trong_khung_hinh': peak,
        'che_do': 'nang_cao' if engine.is_custom else 'co_ban',
    })
    return video_path, report_prefix.with_suffix('.csv'), report_prefix.with_suffix('.json'), summary
