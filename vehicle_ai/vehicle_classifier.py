"""Module nhận diện phương tiện Việt Nam bằng weights YOLO 48 lớp đã huấn luyện.

Lưu weights đã train tại weights/vehicles_vn_best.pt. File weights phải được
train từ dataset có đúng thứ tự lớp trong data/vehicles_vn.yaml.
"""
from __future__ import annotations

from pathlib import Path

from .labels import VN_LABELS


class VehicleClassifier:
    """Wrapper kiểm tra model 48 lớp và giữ giao diện predict của Ultralytics."""

    def __init__(self, weights: str | Path = "weights/vehicles_vn_best.pt"):
        self.weights = Path(weights)
        if not self.weights.is_file():
            raise FileNotFoundError(f"Không tìm thấy model phương tiện: {self.weights}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Chưa cài ultralytics trong môi trường Python hiện tại") from exc
        self.model = YOLO(str(self.weights))
        self.names = self.model.names
        self.class_names = self._normalize_names(self.names)
        expected = set(VN_LABELS)
        actual = set(self.class_names.values())
        self.missing_classes = sorted(expected - actual)
        self.extra_classes = sorted(actual - expected)

    @staticmethod
    def _normalize_names(names) -> dict[int, str]:
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(value) for i, value in enumerate(names)}

    @property
    def is_full_vietnam_model(self) -> bool:
        return not self.missing_classes and not self.extra_classes and len(self.class_names) == 48

    def predict(self, **kwargs):
        """Trả kết quả Ultralytics nguyên bản để engine xử lý bbox/mask."""
        return self.model.predict(**kwargs)

    def validation_message(self) -> str:
        if self.is_full_vietnam_model:
            return "Model có đủ 48 lớp theo danh mục VehicleAI Việt Nam."
        return ("Cảnh báo: weights không khớp đầy đủ 48 lớp. "
                f"Thiếu: {self.missing_classes}; lớp ngoài danh mục: {self.extra_classes}. "
                "Cần kiểm tra lại thứ tự/lớp trong YAML trước khi dùng.")
