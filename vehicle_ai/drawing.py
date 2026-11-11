"""Vẽ tiếng Việt bằng PIL vì cv2.putText không hỗ trợ Unicode đầy đủ."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    locations = [
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for name in locations:
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


def annotate(image_bgr: np.ndarray, vehicles, total: int | None = None) -> np.ndarray:
    canvas = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    painter = ImageDraw.Draw(canvas)
    w, h = canvas.size
    sz = max(14, min(23, int(w/55)))
    font = _font(sz)
    small = _font(max(13, sz-3))
    thick = max(2, round(w / 600))
    for v in vehicles:
        x1, y1, x2, y2 = v.bbox
        x1, x2 = max(0, x1), min(w-1, x2)
        y1, y2 = max(0, y1), min(h-1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        border = (0, 197, 145) if v.confidence >= .55 else (255, 180, 48)
        painter.rectangle((x1, y1, x2, y2), outline=border, width=thick)
        prefix = f'#{v.track_id} ' if v.track_id else ''
        line1 = f'{prefix}{v.label}  {v.confidence:.0%}'
        line2 = f'Màu: {v.color}'
        line3 = f'Biển số: {v.plate_text}' if getattr(v, 'plate_text', '') else ''
        lines = (line1, line2, line3) if line3 else (line1, line2)
        widths = [painter.textbbox((0, 0), line, font=font if i == 0 else small)[2]
                  for i, line in enumerate(lines)]
        bw = min(w-x1, max(widths)+16)
        bh = sz*len(lines) + 20
        top = y1-bh if y1 >= bh else min(h-bh, y2+2)
        if top < 0:
            top = 0
        painter.rectangle((x1, top, x1+bw, top+bh), fill=(15, 29, 41))
        painter.text((x1+7, top+3), line1, fill=(255, 255, 255), font=font)
        painter.text((x1+7, top+sz+9), line2, fill=(193, 235, 226), font=small)
        if line3:
            painter.text((x1+7, top+sz*2+13), line3, fill=(255, 225, 160), font=small)
    if total is not None:
        text = f'Phương tiện trong khung hình: {total}'
        ww = painter.textbbox((0, 0), text, font=font)[2] + 24
        painter.rectangle((5, 5, min(w-3, ww), sz+20), fill=(13, 27, 40))
        painter.text((14, 8), text, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
