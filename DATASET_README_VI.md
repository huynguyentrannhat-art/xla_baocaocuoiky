# Dữ liệu đã tích hợp vào VehicleAI (đọc kỹ)

## 1. Biển số Việt Nam – YOLO detection
`dataset/license_plate/` chứa dữ liệu Roboflow Vietnamese License Plate,
version 8, license MIT theo README đi kèm: 1,296 train, 162 valid, 160 test.
Một lớp `license-plate`; nhãn bounding box YOLO. Tách test độc lập.
Huấn luyện: `python train_plate.py --epochs 100 --imgsz 960 --batch 8 --device 0`
(đổi `--device cpu` nếu không có GPU NVIDIA).

## 2. Phương tiện – phân loại ảnh 4 lớp
`dataset/vehicle_classification/` chứa 7,706 ảnh train gốc được chia thành
train/val (90/10) và 2,294 ảnh test giữ riêng; tổng 10,000 ảnh.
Bốn lớp thực tế: car, bus, truck, van. Đây là ảnh crop/phân loại, không có
bounding box. Vì vậy KHÔNG thể dùng chúng để huấn luyện detector YOLO 48 lớp
hoặc tự suy diễn 48 loại xe. Dùng script phân loại riêng:
`python train_vehicle_coarse.py --epochs 80 --imgsz 224 --batch 32 --device 0`.
Kết quả: `weights/vehicles_vn_coarse_cls.pt`.

## Giới hạn và khuyến nghị
- Dataset phương tiện hiện có không chứa xe máy, xe đạp, xe chuyên dụng và
  không đủ chi tiết để chia sedan/SUV, số chỗ, tải trọng... Nó chỉ hỗ trợ
  phân loại thô 4 nhóm.
- Muốn huấn luyện detector 48 lớp, cần ảnh giao thông Việt Nam có bounding
  box và nhãn đúng từng lớp. Không biến nhãn phân loại thành bbox giả.
- Dữ liệu biển số chỉ phát hiện vùng biển; OCR đọc ký tự là công đoạn riêng.
- Dữ liệu test không dùng trong huấn luyện.
