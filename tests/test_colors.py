import unittest
import cv2
import numpy as np

from vehicle_ai.colors import get_vehicle_color
from vehicle_ai.labels import COCO_BASELINE, VN_LABELS
from vehicle_ai.tracker import iou
from vehicle_ai.plate import PlateReader


class ColorTests(unittest.TestCase):
    def get(self, bgr):
        image = np.zeros((220, 330, 3), np.uint8)
        image[:] = (45, 220, 45)  # nền xanh
        image[30:190, 40:290] = bgr
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[30:190, 40:290] = 255
        return get_vehicle_color(image, (40, 30, 290, 190), mask)

    def test_red(self):
        self.assertEqual(self.get((0, 0, 240)).name, 'Đỏ')

    def test_blue(self):
        self.assertEqual(self.get((220, 70, 30)).name, 'Xanh dương')

    def test_white(self):
        self.assertEqual(self.get((245, 245, 245)).name, 'Trắng')

    def test_black(self):
        self.assertEqual(self.get((28, 28, 28)).name, 'Đen')

    def test_multicolor_vehicle_returns_unknown(self):
        image = np.zeros((220, 330, 3), np.uint8)
        image[:] = (45, 220, 45)
        image[30:190, 40:165] = (0, 0, 240)
        image[30:190, 165:290] = (240, 0, 0)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[30:190, 40:290] = 255
        result = get_vehicle_color(image, (40, 30, 290, 190), mask)
        self.assertIn('Không rõ', result.name)

    def test_small_unknown(self):
        image = np.zeros((10, 10, 3), np.uint8)
        self.assertEqual(get_vehicle_color(image, (0, 0, 10, 10)).name, 'Không rõ')

    def test_taxonomy(self):
        self.assertIn('minibus_16seat', VN_LABELS)
        self.assertIn('bus_32seat', VN_LABELS)
        self.assertIn('container_truck', VN_LABELS)
        self.assertIn('fire_truck', VN_LABELS)
        self.assertNotIn('minibus_16seat', COCO_BASELINE)

    def test_vietnam_plate_format(self):
        self.assertTrue(PlateReader._looks_like_vietnam_plate('30A12345'))
        self.assertTrue(PlateReader._looks_like_vietnam_plate('51F12345'))
        self.assertFalse(PlateReader._looks_like_vietnam_plate('ABCDE'))
        self.assertFalse(PlateReader._looks_like_vietnam_plate('12345678'))

    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 100, 100), (0, 0, 100, 100)), 1)
        self.assertAlmostEqual(iou((0, 0, 100, 100), (200, 200, 300, 300)), 0)


if __name__ == '__main__':
    unittest.main()
