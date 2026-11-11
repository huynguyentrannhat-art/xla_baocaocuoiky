"""Mock YOLO giúp test luồng xử lý không cần tải .pt."""
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from vehicle_ai.engine import VehicleEngine
from vehicle_ai.runner import imwrite_unicode, run_image
from vehicle_ai.tracker import IoUTracker


class MockBox:
    def __init__(self, bbox, cls=2, conf=.92):
        self.xyxy = np.array([bbox], dtype=float)
        self.cls = np.array(cls)
        self.conf = np.array(conf)


class MockMasks:
    def __init__(self, polygon):
        self.xy = [np.array(polygon, dtype=np.float32)]


class MockResults:
    names = {2: 'car'}
    def __init__(self):
        self.boxes = [MockBox((35, 30, 285, 185))]
        self.masks = MockMasks([(35, 30), (285, 30), (285, 185), (35, 185)])


class PipelineTests(unittest.TestCase):
    def test_full_image_flow_mock_ai(self):
        frame = np.full((210, 320, 3), (0, 255, 0), dtype=np.uint8)  # nền
        cv2.rectangle(frame, (35, 30), (285, 185), (0, 0, 230), -1)  # xe giả lập đỏ
        engine = VehicleEngine.__new__(VehicleEngine)
        engine.is_custom = False
        engine.detector = object()
        engine.segmenter = None
        engine.tracker = IoUTracker()
        engine._predict = lambda model, image: MockResults()
        result = engine.process(frame, tracking=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].color, 'Đỏ')
        self.assertEqual(result[0].track_id, 1)
        self.assertIn('Ô tô con', result[0].label)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / 'xe_việt_nam.png'
            imwrite_unicode(src, frame)
            saved, csv_path, json_path, detections, summary = run_image(engine, src, Path(tmp) / 'out')
            self.assertTrue(saved.exists())
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(len(detections), 1)
            self.assertEqual(summary['so_luot_phat_hien'], 1)
            self.assertIn('Ô tô con', csv_path.read_text(encoding='utf-8-sig'))


if __name__ == '__main__':
    unittest.main()
