# Dataset ban vua cho bao cao VehicleAI

## Quy mo
- Phuong tien: 1680 anh, 4 lop.
- Bien so: 750 anh co cap nhan YOLO.
- Tong cong: 2430 anh.

## Phuong tien
Moi lop bus/car/truck/van:
- Train: 280 anh.
- Validation: 70 anh.
- Test doc lap: 70 anh.
Tong 1.680 anh. Anh train va validation duoc lay tu tap train goc; test lay tu tap test goc.
Day la phan loai anh theo thu muc lop, khong phai detector bounding-box va khong dai dien 48 lop xe.

## Bien so
- Train: 500 anh.
- Validation: 125 anh.
- Test: 125 anh.
- Anh va nhan YOLO duoc giu theo cap; data.yaml da sua duong dan tu thu muc goc du an.

## Lenh chay tren CPU
python train_plate.py --epochs 30 --imgsz 640 --batch 2 --device cpu
python train_vehicle_coarse.py --epochs 30 --imgsz 224 --batch 8 --device cpu

Luu y: Day la tap con vua phai lay mau tu cac ZIP nguon nguoi dung cung cap. Ket qua bao cao can ghi ro quy mo va nguon; khong tuyen bo do chinh xac tong quat khi chua danh gia tren tap test.
