"""Pipeline: YOLO26-seg (COCO) + [YOLO tùy biến] + HSV + gán ID IoU."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .colors import get_vehicle_color
from .labels import BROAD_GROUPS, COCO_BASELINE, VN_LABELS
from .tracker import IoUTracker, iou
from .plate import PlateReader
from .vehicle_classifier import VehicleClassifier


@dataclass
class Vehicle:
    bbox: tuple[int, int, int, int]
    label: str
    class_key: str
    confidence: float
    color: str
    color_confidence: float
    color_method: str
    track_id: int | None = None
    plate_text: str = ""
    plate_confidence: float = 0.0
    plate_status: str = "Chưa cấu hình model biển số"

    def to_dict(self):
        x1, y1, x2, y2 = self.bbox
        return {
            'track_id': self.track_id or '', 'loai_xe': self.label,
            'ma_lop': self.class_key, 'do_tin_cay_xe': round(self.confidence, 4),
            'mau_xe': self.color, 'do_tin_cay_mau': self.color_confidence,
            'cach_lay_mau': self.color_method,
            'bien_so': self.plate_text,
            'do_tin_cay_bien_so': self.plate_confidence,
            'trang_thai_bien_so': self.plate_status,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        }


def _name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    return str(names[class_id])


def _polygon_mask(result, index: int, shape: tuple[int, int]) -> np.ndarray | None:
    # Ultralytics masks.xy trả tọa độ trên ảnh gốc, thích hợp với ảnh bất kỳ cỡ nào.
    masks = getattr(result, 'masks', None)
    if masks is None or len(masks.xy) <= index:
        return None
    poly = np.asarray(masks.xy[index], dtype=np.int32)
    if len(poly) < 3:
        return None
    output = np.zeros(shape, np.uint8)
    cv2.fillPoly(output, [poly], 255)
    return output


def _results_objects(result, image_shape):
    if result is None or getattr(result, 'boxes', None) is None:
        return []
    objects = []
    for index, box in enumerate(result.boxes):
        key = _name(result.names, int(box.cls.item()))
        x1, y1, x2, y2 = [int(round(float(v))) for v in box.xyxy[0].tolist()]
        x1 = max(0, min(image_shape[1], x1))
        x2 = max(0, min(image_shape[1], x2))
        y1 = max(0, min(image_shape[0], y1))
        y2 = max(0, min(image_shape[0], y2))
        if x2-x1 < 9 or y2-y1 < 9:
            continue
        objects.append({
            'bbox': (x1, y1, x2, y2), 'key': key,
            'conf': float(box.conf.item()),
            'mask': _polygon_mask(result, index, image_shape),
        })
    return objects


class VehicleEngine:
    def __init__(self, custom_weights: str | None = None, conf: float = 0.38,
                 imgsz: int = 768, device: str | None = None, use_seg: bool = True,
                 base_weights: str = 'yolo26n-seg.pt',
                 plate_weights: str = 'weights/license_plate_detector.pt',
                 enable_plate: bool = True):
        if not (0.05 <= conf <= .95):
            raise ValueError('conf phải trong khoảng 0.05..0.95')
        from ultralytics import YOLO
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        # Có hai LOẠI weights khác nhau: 48-class detection và 4-class classification.
        # Classification không có boxes, nên phải tìm bbox bằng COCO trước, rồi
        # phân loại lại ảnh crop theo car/bus/truck/van.
        self.is_custom = False
        self.vehicle_classifier = None
        self.coarse_classifier = None
        if custom_weights:
            if not Path(custom_weights).is_file():
                raise FileNotFoundError(f'Chưa có mô hình riêng: {custom_weights}')
            candidate = YOLO(str(custom_weights))
            if getattr(candidate, 'task', '') == 'classify':
                self.coarse_classifier = candidate
                self.detector = YOLO(base_weights)
                self.segmenter = None
            else:
                self.is_custom = True
                self.vehicle_classifier = VehicleClassifier(custom_weights)
                self.detector = self.vehicle_classifier
                self.segmenter = YOLO(base_weights) if use_seg else None
        else:
            self.detector = YOLO(base_weights)
            self.segmenter = None
        self.tracker = IoUTracker()
        self.plate_reader = PlateReader(plate_weights) if enable_plate else None

    def _predict(self, model, image):
        kw = {'source': image, 'conf': self.conf, 'imgsz': self.imgsz,
              'verbose': False, 'retina_masks': True}
        if self.device:
            kw['device'] = self.device
        return model.predict(**kw)[0]

    def process(self, image_bgr: np.ndarray, tracking: bool = False) -> list[Vehicle]:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError('Cần ảnh BGR 3 kênh')
        shape = image_bgr.shape[:2]
        detected = _results_objects(self._predict(self.detector, image_bgr), shape)
        segmented = []
        if self.is_custom and self.segmenter is not None and detected:
            segmented = _results_objects(self._predict(self.segmenter, image_bgr), shape)
            segmented = [s for s in segmented if s['key'] in COCO_BASELINE]

        vehicles = []
        for d in detected:
            key = d['key']
            if self.is_custom:
                # Không tự đổi nhãn; class không nằm trong YAML được đánh dấu rõ.
                if key not in VN_LABELS:
                    continue
                label = VN_LABELS[key]
                # Vật thể quá nhỏ: tránh khẳng định cụ thể số chỗ hoặc kiểu xe.
                x1, y1, x2, y2 = d['bbox']
                if x2 - x1 < 82 or y2 - y1 < 52:
                    group = BROAD_GROUPS.get(key)
                    label = (COCO_BASELINE[group][0] if group in COCO_BASELINE
                             else 'Phương tiện (quá nhỏ để phân loại chi tiết)')
            else:
                if key not in COCO_BASELINE:
                    continue
                label = COCO_BASELINE[key][0]
                # Phân loại 4 nhóm xe trong bbox do YOLO COCO tìm được.
                # Không áp dụng classifier ô tô cho bicycle/motorcycle.
                if getattr(self, 'coarse_classifier', None) is not None and key in {'car', 'bus', 'truck'}:
                    x1, y1, x2, y2 = d['bbox']
                    if x2 - x1 >= 40 and y2 - y1 >= 40:
                        crop = image_bgr[y1:y2, x1:x2]
                        try:
                            res = self.coarse_classifier.predict(source=crop, imgsz=224,
                                                                 verbose=False)[0]
                            probs = res.probs
                            if probs is not None:
                                cid = int(probs.top1)
                                cconf = float(probs.top1conf.item())
                                coarse_key = str(res.names[cid]).lower()
                                group_labels = {
                                    'car': 'Ô tô con (phân loại 4 nhóm)',
                                    'bus': 'Xe buýt / xe khách (phân loại 4 nhóm)',
                                    'truck': 'Xe tải (phân loại 4 nhóm)',
                                    'van': 'Xe van (phân loại 4 nhóm)',
                                }
                                if cconf >= .55 and coarse_key in group_labels:
                                    label = group_labels[coarse_key]
                        except Exception:
                            # Model phân loại phụ trợ lỗi không làm mất kết quả COCO.
                            pass

            mask = d['mask']
            if self.is_custom and segmented:
                # Chỉ ghép đúng nhóm COCO hoặc cho phép kiểm tra IoU thật cao
                # (vì xe cứu thương/van có thể được COCO xem là bus hoặc car).
                group = BROAD_GROUPS.get(key)
                compatible = [(iou(d['bbox'], s['bbox']), s) for s in segmented
                              if s['key'] == group]
                if not compatible:
                    compatible = [(iou(d['bbox'], s['bbox']), s) for s in segmented]
                if compatible:
                    best_iou, best_seg = max(compatible, key=lambda pair: pair[0])
                    threshold = 0.28 if best_seg['key'] == group else 0.72
                    if best_iou >= threshold:
                        mask = best_seg['mask']

            color = get_vehicle_color(image_bgr, d['bbox'], mask)
            vehicles.append(Vehicle(
                bbox=d['bbox'], label=label, class_key=key,
                confidence=round(d['conf'], 3), color=color.name,
                color_confidence=color.confidence, color_method=color.method,
            ))

        # Module biển số chạy độc lập; không đọc được thì để trống, không suy đoán.
        if getattr(self, "plate_reader", None) is not None:
            self.plate_reader.attach(image_bgr, vehicles)

        if tracking:
            self.tracker.update(vehicles)
        return vehicles
