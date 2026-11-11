"""Ước lượng màu sơn thân xe bằng HSV trên vùng đối tượng.

Màu nhận từ ảnh gốc (KHÔNG CLAHE/chỉnh gamma để tránh đổi màu).
Nếu có instance mask, mẫu màu lấy bên trong mask thay vì từ cả nền.
Không phân biệt màu sơn và decal đa sắc bằng một nhãn đơn.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorResult:
    name: str
    confidence: float
    samples: int
    method: str


def _names_for_hsv(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    names = np.full(len(h), 'khac', dtype='<U12')
    low_sat = s < 52
    names[low_sat & (v < 73)] = 'den'
    names[low_sat & (v >= 73) & (v < 145)] = 'xam'
    names[low_sat & (v >= 145) & (v < 218)] = 'bac'
    names[low_sat & (v >= 218)] = 'trang'

    # Mức sáng rất thấp: không cố gắng gọi đỏ/xanh dựa vào nhiễu màu.
    names[v < 45] = 'den'
    chroma = ~low_sat & (v >= 45)
    names[chroma & ((h < 9) | (h >= 174))] = 'do'
    names[chroma & (h >= 9) & (h < 19)] = 'cam'
    names[chroma & (h >= 19) & (h < 36)] = 'vang'
    names[chroma & (h >= 36) & (h < 88)] = 'xanh_la'
    names[chroma & (h >= 88) & (h < 102)] = 'xanh_ngoc'
    names[chroma & (h >= 102) & (h < 133)] = 'xanh_duong'
    names[chroma & (h >= 133) & (h < 155)] = 'tim'
    names[chroma & (h >= 155) & (h < 174)] = 'hong'
    # Màu tối xỉn có thể là bóng râm; tránh tự đoán là nâu nếu không đủ tin cậy.
    names[chroma & (h >= 5) & (h < 24) & (v < 156) & (s < 190)] = 'nau'
    return names


COLOR_NAMES = {
    'den': 'Đen', 'xam': 'Xám', 'bac': 'Bạc', 'trang': 'Trắng',
    'do': 'Đỏ', 'cam': 'Cam', 'vang': 'Vàng', 'xanh_la': 'Xanh lá',
    'xanh_ngoc': 'Xanh ngọc', 'xanh_duong': 'Xanh dương',
    'tim': 'Tím', 'hong': 'Hồng', 'nau': 'Nâu',
}


def get_vehicle_color(frame: np.ndarray, bbox: tuple[int, int, int, int],
                      object_mask: np.ndarray | None = None) -> ColorResult:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    w, h = x2 - x1, y2 - y1
    if w < 36 or h < 28:
        return ColorResult('Không rõ', 0., 0, 'quá nhỏ')

    # Hạn chế mẫu ở phần thân, tránh mép có nền, giảm kính & bánh xe.
    yy, xx = np.ogrid[:h, :w]
    core = ((xx >= int(.08*w)) & (xx < int(.92*w)) &
            (yy >= int(.22*h)) & (yy < int(.86*h)))
    method = 'hình chữ nhật'
    if object_mask is not None and object_mask.shape == frame.shape[:2]:
        mask = (object_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        k = max(3, (min(w, h) // 45) | 1)
        mask = cv2.erode(mask, np.ones((k, k), np.uint8), iterations=1)
        core = core & (mask > 0)
        method = 'mặt nạ phân vùng'
    else:
        # BBox thường chứa cả nền, giới hạn vùng mẫu thấp hơn.
        core = core & (yy >= int(.43*h)) & (yy < int(.80*h)) & \
                    (xx >= int(.16*w)) & (xx < int(.84*w))

    pixels = frame[y1:y2, x1:x2][core]
    if len(pixels) < 130:
        return ColorResult('Không rõ', 0., len(pixels), method)
    # Giảm chi phí khi xe chiếm quá nhiều pixel.
    if len(pixels) > 60000:
        pixels = pixels[::int(np.ceil(len(pixels)/60000))]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # Bỏ phần tối (kính, lốp và bóng đổ) khi có đủ màu sáng để so sánh.
    dark = hsv[:, 2] < 75
    if 0.10 < float(np.mean(dark)) < 0.48 and np.count_nonzero(~dark) >= 150:
        hsv = hsv[~dark]
    labels = _names_for_hsv(hsv)
    names, counts = np.unique(labels[labels != 'khac'], return_counts=True)
    if not len(counts):
        return ColorResult('Không rõ', 0., len(hsv), method)
    order = np.argsort(counts)[::-1]
    chosen, best = str(names[order[0]]), int(counts[order[0]])
    support = best / max(1, len(hsv))

    # Nếu hai màu cùng chiếm tỷ lệ lớn, không ép thành một màu duy nhất.
    # Điều này giảm nhầm xe sơn 2 màu/decal lớn thành màu trội giả.
    if len(order) >= 2:
        second = int(counts[order[1]])
        second_ratio = second / max(1, best)
        top_two = (best + second) / max(1, len(hsv))
        if second_ratio >= 0.65 and top_two >= 0.60:
            return ColorResult('Không rõ (xe nhiều màu)', round(top_two, 3),
                               len(hsv), method)

    if support < 0.30:
        return ColorResult('Không rõ (nhiều màu/ánh sáng khó)', round(support, 3), len(hsv), method)
    return ColorResult(COLOR_NAMES[chosen], round(float(support), 3), len(hsv), method)
