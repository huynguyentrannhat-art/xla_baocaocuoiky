"""Nhận diện và đọc biển số Việt Nam (tùy chọn, cần model biển số riêng).

Đặt weights/license_plate_detector.pt (YOLO detect biển số) và cài
Tesseract OCR + `pip install pytesseract`. Nếu thiếu một trong các thành phần,
module bỏ qua an toàn, không tự bịa biển số.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PlateResult:
    text: str = ""
    confidence: float = 0.0
    status: str = "Chưa cấu hình model biển số"


class PlateReader:
    def __init__(self, weights: str | Path = "weights/license_plate_detector.pt",
                 conf: float = 0.30, min_ocr_conf: float = 55.0):
        self.weights = Path(weights)
        self.conf = conf
        self.min_ocr_conf = min_ocr_conf
        self.detector = None
        self.ocr = None
        self.status = "Chưa có weights/license_plate_detector.pt"
        if self.weights.is_file():
            try:
                from ultralytics import YOLO
                self.detector = YOLO(str(self.weights))
                try:
                    import os
                    import pytesseract
                    self.ocr = pytesseract
                    command = os.environ.get("TESSERACT_CMD", "").strip()
                    if command:
                        self.ocr.pytesseract.tesseract_cmd = command
                    self.ocr.get_tesseract_version()
                    self.status = "Sẵn sàng: phát hiện biển số + OCR"
                except Exception:
                    self.ocr = None
                    self.status = "Model biển số đã tải; chưa cài/không tìm thấy Tesseract OCR"
            except Exception as exc:
                self.status = f"Không tải được model biển số: {exc}"

    @property
    def enabled(self) -> bool:
        # Detector vẫn có thể tìm khung biển số dù Tesseract chưa cài.
        return self.detector is not None

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def _looks_like_vietnam_plate(text: str) -> bool:
        # Mẫu tổng quát cho biển dân sự phổ biến; không bao phủ mọi biển đặc thù.
        return bool(re.fullmatch(r"\d{2}[A-Z]{1,2}\d{4,5}", text))

    def _read_crop(self, crop: np.ndarray) -> PlateResult:
        if crop.size == 0 or min(crop.shape[:2]) < 18:
            return PlateResult(status="Biển số quá nhỏ")
        h, w = crop.shape[:2]
        if w < 90 or h < 22 or w / max(h, 1) < 1.7:
            return PlateResult(status="Vùng biển số quá nhỏ/méo")
        scale = max(2, min(4, 320 // max(w, 1)))
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 7, 45, 45)
        # Thử ảnh xám và nhị phân; chỉ nhận kết quả OCR có cấu trúc biển hợp lệ.
        variants = [gray, cv2.threshold(gray, 0, 255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
        best = PlateResult(status="OCR không đọc rõ")
        # Thử nhiều chế độ vì biển Việt Nam có loại 1 dòng và 2 dòng.
        for psm in (7, 6, 11):
            config = ("--oem 3 --psm " + str(psm) +
                      " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            for variant in variants:
                try:
                    data = self.ocr.image_to_data(
                        variant, config=config, output_type=self.ocr.Output.DICT)
                    tokens, scores = [], []
                    for raw, score in zip(data.get("text", []), data.get("conf", [])):
                        token = self._normalize(raw)
                        try:
                            confidence = float(score)
                        except (TypeError, ValueError):
                            continue
                        if token and confidence >= self.min_ocr_conf:
                            tokens.append(token)
                            scores.append(confidence)
                    candidate = "".join(tokens)
                    if candidate and self._looks_like_vietnam_plate(candidate) and scores:
                        confidence = sum(scores) / len(scores)
                        if confidence > best.confidence:
                            best = PlateResult(text=candidate, confidence=confidence,
                                               status="Đọc được")
                except Exception:
                    continue
        return best

    def attach(self, frame: np.ndarray, vehicles: list) -> None:
        """Detect một lần trên frame, gán biển cho xe chứa vùng biển số."""
        if not vehicles:
            return
        if not self.enabled:
            for vehicle in vehicles:
                vehicle.plate_status = self.status
            return
        try:
            result = self.detector.predict(frame, conf=self.conf, imgsz=960,
                                           verbose=False)[0]
        except Exception:
            for vehicle in vehicles:
                vehicle.plate_status = "Lỗi model biển số"
            return
        plates = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = [int(round(float(x))) for x in box.xyxy[0].tolist()]
                x1, x2 = max(0, x1), min(frame.shape[1], x2)
                y1, y2 = max(0, y1), min(frame.shape[0], y2)
                if x2 - x1 >= 50 and y2 - y1 >= 16:
                    plates.append((x1, y1, x2, y2))
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            candidates = []
            for plate_box in plates:
                px1, py1, px2, py2 = plate_box
                cx, cy = (px1 + px2) / 2, (py1 + py2) / 2
                if vx1 <= cx <= vx2 and vy1 <= cy <= vy2:
                    area = (px2-px1) * (py2-py1)
                    candidates.append((area, plate_box))
            if not candidates:
                vehicle.plate_status = "Không tìm thấy vùng biển số"
                continue
            _, (x1, y1, x2, y2) = max(candidates, key=lambda item: item[0])
            # Nới nhẹ crop nhưng không vượt ra ngoài ảnh.
            pad_x, pad_y = int((x2-x1)*.06), int((y2-y1)*.12)
            crop = frame[max(0,y1-pad_y):min(frame.shape[0],y2+pad_y),
                         max(0,x1-pad_x):min(frame.shape[1],x2+pad_x)]
            if self.ocr is None:
                vehicle.plate_status = "Đã phát hiện vùng biển số; cần cài Tesseract OCR để đọc chữ"
                continue
            read = self._read_crop(crop)
            vehicle.plate_text = read.text
            vehicle.plate_confidence = round(read.confidence / 100, 3)
            vehicle.plate_status = read.status
