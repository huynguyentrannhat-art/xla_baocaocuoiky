# BAT DAU O DAY – VEHICLEAI VIET NAM (VS CODE)

> HƯỚNG DẪN WINDOWS 10/11, PYTHON 3.14 / 3.11+. Không phải mở file .md để chạy AI.
> Lần chạy đầu tiên cần Internet để tải model `yolo26n-seg.pt` (~vài MB đến vài chục MB tùy model).

## 1. Cài Python 3.14 (hoặc 3.11+)

1. Mở https://www.python.org/downloads/ và tải Python **3.11 bản 64-bit cho Windows**.
2. Chạy file cài, tích **Add Python to PATH** (nếu installer có lựa chọn này), chọn **Install Now**.
3. Đóng/mở lại VS Code; vào `Terminal > New Terminal`.
4. Kiểm tra:

```powershell
py -3.14 --version
py --list
```

Kỳ vọng có dòng `Python 3.11.x`. Nếu thiếu, cài Python 3.11 trước.

## 2. Cài Visual Studio Code

1. Tải VS Code tại https://code.visualstudio.com/.
2. Mở VS Code -> Extensions (`Ctrl+Shift+X`) -> cài **Python** của Microsoft.
3. Giải nén ZIP dự án vào thư mục NGẮN, ví dụ `D:\VehicleAI_VN`.
4. VS Code -> File -> Open Folder -> chọn **thư mục chứa main.py**.
5. Mở Terminal (`Ctrl+``) -> chọn **PowerShell**.

## 3. Tạo môi trường và cài thư viện

Gõ lần lượt trong Terminal đang ở thư mục dự án:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**Không cần kích hoạt** môi trường, tránh lỗi PowerShell `running scripts is disabled` / `ExecutionPolicy`.

Kiểm tra:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import cv2, ultralytics, streamlit; print('CAI DAT THANH CONG')"
```

**Dùng phím chọn Python ở VS Code:** `Ctrl+Shift+P` -> `Python: Select Interpreter` -> chọn `.venv\Scripts\python.exe`.

> Cách đơn giản thay thế: nháy đúp `1_CAI_DAT.bat` trong thư mục dự án; sau đó chạy lệnh dưới đây hoặc `2_MO_GIAO_DIEN.bat`.

## 4. Chạy giao diện (khuyên dùng)

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Sau đó mở trình duyệt `http://localhost:8501` nếu không tự mở. Chọn Ảnh, Webcam (chụp ảnh), hoặc Video, rồi bấm **Bắt đầu nhận diện**.

> Webcam trong Streamlit dùng **chụp ảnh**. Muốn camera thời gian thực hãy dùng mục 5.

## 5. Chạy camera thời gian thực

```powershell
.\.venv\Scripts\python.exe main.py --source 0
```

Nhấn **Q** trong cửa sổ camera để dừng. Nếu báo `Không mở được nguồn 0`:

```powershell
.\.venv\Scripts\python.exe main.py --source 1
```

Kiểm tra Windows `Settings > Privacy & security > Camera`; đóng Zoom/Teams/ứng dụng đang dùng camera.

## 6. Chạy ảnh từ thư mục máy

Đặt ảnh ví dụ `D:\VehicleAI_VN\xe.jpg` rồi chạy:

```powershell
.\.venv\Scripts\python.exe main.py --source "xe.jpg"
```

Hoặc đường dẫn đầy đủ:

```powershell
.\.venv\Scripts\python.exe main.py --source "D:\AnhGiaoThong\anh1.jpg"
```

Kết quả lưu ở `output/anh1_ketqua.png`, `output/anh1_thongke.csv`, `output/anh1_thongke.json`.

## 7. Chạy video

```powershell
.\.venv\Scripts\python.exe main.py --source "video.mp4" --show
```

Kết quả: `output/video_ketqua.mp4`, CSV, JSON. Nếu mở video không được trong trình duyệt, tải file xuống và mở bằng VLC.

## 8. Chọn tốc độ và mức độ tin cậy

```powershell
.\.venv\Scripts\python.exe main.py --source 0 --imgsz 640 --conf 0.45
```

- `--imgsz 640`: nhanh hơn, dễ bỏ xe nhỏ ở xa.
- `--imgsz 960` hoặc `1280`: có thể tốt hơn cho xe xa nhưng chậm hơn.
- `--conf 0.45`: lọc những dự đoán thấp hơn 45%; không đồng nghĩa với độ chính xác thực 45%.
- `--device cpu`: chạy CPU. `--device 0`: dùng GPU CUDA nếu PyTorch nhận GPU tương thích.
- `--max-frames 100`: chạy thử 100 khung hình, hữu ích khi kiểm tra.

## 9. Vì sao chưa nhận ra xe khách 16, 32 chỗ hay cứu hỏa?

Vì bản tải model có sẵn chỉ biết 5 nhóm thuộc COCO: `bicycle`, `motorcycle`, `car`, `bus`, `truck`. Đây là tính năng **chạy ngay**. Muốn nhận diện chi tiết 48 loại theo Việt Nam, **bắt buộc cần dữ liệu ảnh có gán nhãn + huấn luyện**. Các file bài giảng PDF không chứa trọng số mô hình nhận diện 48 lớp.

Các tên chi tiết có trong `data/vehicles_vn.yaml` và `vehicle_ai/labels.py`. Các tên là *danh mục đào tạo*, không có nghĩa mô hình đã học được.

## 10. Cách tự huấn luyện mô hình 48 loại

### Bước 10.1 – Chuẩn bị ảnh

Bỏ ảnh vào đúng cấu trúc:

```text
dataset/
  images/
    train/
      xe001.jpg
      xe002.jpg
    val/
      xe101.jpg
  labels/
    train/
      xe001.txt
      xe002.txt
    val/
      xe101.txt
```

Các cặp `.jpg` và `.txt` phải **trùng tên gốc**. Chọn `train` 80% và `val` 20% ở mức tập ảnh; không đưa khung gần trùng từ cùng video vào cả hai tập. Nên có **tập test độc lập** sau khi train. Với 48 loại, cần thu thập rất nhiều ảnh từ nhiều góc/ánh sáng/địa điểm; khoảng vài trăm đến vài nghìn ví dụ cho **mỗi loại** là mục tiêu dữ liệu hợp lý để thử nghiệm, không phải cam kết kết quả. Chú ý nhãn số ghế phải xác thực theo xe, không đoán từ vỏ xe.

### Bước 10.2 – Gán nhãn trực tiếp trên máy

Chạy:

```powershell
.\.venv\Scripts\python.exe label_dataset.py --split train
.\.venv\Scripts\python.exe label_dataset.py --split val
```

Chương trình in ra **mã số 48 lớp**. Nhập mã lớp bằng terminal. Khi cửa sổ ảnh hiện ra:

- **Kéo chuột trái** để khoanh vùng từng xe.
- **C**: đổi mã lớp (quay lại terminal nhập số mới).
- **Z**: hoàn tác khung vừa vẽ.
- **S**: lưu file `.txt` và chuyển ảnh kế tiếp.
- **N**: bỏ qua ảnh (không lưu).
- **Q**: thoát.

Chú ý nhãn xe cứu hỏa/xe cứu thương dựa vào dấu hiệu nhận biết thực tế; nhãn xe 16/32 chỗ phải có thông tin xác nhận, không dựa thuần vào hình ảnh. Nếu xe không xác định được lớp, đừng cố gán nhãn sai; cần loại khỏi tập huấn luyện cho đến khi xác định rõ.

Định dạng một dòng nhãn YOLO:

```text
class_id x_center y_center width height
```

Các tọa độ nằm trong khoảng 0..1. Công cụ `label_dataset.py` tự tạo định dạng này.

### Bước 10.3 – Kiểm tra bộ dữ liệu

```powershell
.\.venv\Scripts\python.exe check_dataset.py
```

Nếu hiện “Không có ảnh”, là **đúng** khi mới giải nén (ZIP không chứa dataset của 48 loại).

### Bước 10.4 – Train

Để thử trên máy chưa có GPU:

```powershell
.\.venv\Scripts\python.exe train.py --base yolo26n.pt --epochs 30 --imgsz 640 --batch 4 --device cpu
```

Khi có GPU NVIDIA đã cài PyTorch CUDA phù hợp:

```powershell
.\.venv\Scripts\python.exe train.py --base yolo26s.pt --epochs 100 --imgsz 960 --batch 8 --device 0
```

Đầu ra: `runs/vehicles_vn/weights/best.pt`. Script tự sao chép sang `weights/vehicles_vn_best.pt`.

> **Không có dataset thực tế thì train sẽ dừng** thay vì tự tạo số liệu hoặc tự nhận xe sai. Huấn luyện 48 lớp có thể tốn thời gian đáng kể, đặc biệt bằng CPU.

### Bước 10.5 – Dùng model đã train

Sau khi có `weights/vehicles_vn_best.pt`, chạy giao diện/camera như trên. Chương trình **tự nhận biết** model riêng và chuyển chế độ chi tiết.

Hoặc chỉ định trọng số:

```powershell
.\.venv\Scripts\python.exe main.py --source "xe.jpg" --weights "weights/vehicles_vn_best.pt"
```

### Bước 10.6 – Đánh giá (không được tự công bố accuracy)

```powershell
.\.venv\Scripts\python.exe evaluate.py --split val --device cpu
```

Xem mAP50, mAP50-95, confusion matrix, precision/recall và kiểm tra nhầm lẫn giữa `car/bus/truck`, 7/16/32 chỗ, `container_truck` và `heavy_truck`. Tập val và test không được trùng train.

## 11. Khắc phục các lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `py -3.11` not found | Cài Python 3.14 (hoặc 3.11+) 64-bit, đóng/mở lại terminal |
| `python` hoặc `pip` không nhận | Luôn gọi `.\.venv\Scripts\python.exe -m pip ...` |
| `npm.ps1`/`Activate.ps1` blocked | Không cần Activate.ps1; dùng trực tiếp python.exe trong .venv |
| `No module named ultralytics` | Cài lại `-m pip install -r requirements.txt` bằng đúng python.exe |
| `torch.cuda.is_available() == False` | Đang chạy CPU; cài PyTorch CUDA đúng theo https://pytorch.org/get-started/locally/ nếu có NVIDIA |
| `Camera index out of range` | Thử `--source 0`, không mặc định 1; đóng ứng dụng đang dùng webcam |
| Model tải `.pt` thất bại | Kiểm tra Internet, proxy, antivirus; thử lại |
| Màu không đúng | Dùng ảnh gốc rõ, không chỉnh màu mạnh; bật phân vùng (`use_seg`) |
| Xe 16 chỗ được gọi 'Xe buýt/xe khách' | Đang dùng COCO cơ bản; cần model riêng |
| `Trained model` sai loại | Tăng ảnh gán nhãn, rà nhầm nhãn, train và kiểm tra ma trận nhầm lẫn |
| `No module named streamlit` | Cài `requirements.txt`, hoặc chỉ chạy `main.py` |
| `cv2.imshow` không mở cửa sổ | Chạy tại Windows có desktop; không chạy trong môi trường không có GUI |
| `VideoWriter` báo lỗi | Kiểm tra OpenCV/codec, thử clip MP4 khác, dùng VLC để xem file |

## 12. Kiểm thử mã nguồn không cần tải AI

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bộ test này kiểm tra nhận diện màu trên ảnh tổng hợp và cấu trúc tên xe; **không phải** đo độ chính xác AI trên dữ liệu xe thực tế.

## 13. File nào để sửa?

- `vehicle_ai/labels.py`: 48 nhãn tiếng Việt và nhãn COCO cơ bản.
- `data/vehicles_vn.yaml`: thứ tự lớp train; KHÔNG thay thứ tự sau khi đã train nếu không train lại.
- `vehicle_ai/colors.py`: các ngưỡng HSV, lấy màu theo mask.
- `vehicle_ai/engine.py`: lựa chọn YOLO và ghép bbox-mask.
- `vehicle_ai/drawing.py`: vẽ khung và chữ tiếng Việt.
- `main.py`: lệnh ảnh/video/camera.
- `app.py`: giao diện trình duyệt.
- `label_dataset.py`, `train.py`, `evaluate.py`: dữ liệu và huấn luyện.

Đọc `README.md` để hiểu kiến trúc, yêu cầu dữ liệu và giới hạn kỹ thuật.


## 11. Nhận diện biển số (tùy chọn, chưa bật nếu thiếu model)

Module `vehicle_ai/plate.py` đã được bổ sung nhưng KHÔNG có sẵn model biển số trong ZIP.
Để bật tính năng:
1. Có model YOLO detect biển số đã huấn luyện phù hợp ảnh giao thông Việt Nam.
2. Đặt đúng tên `weights/license_plate_detector.pt`.
3. Cài Tesseract OCR cho Windows và thêm `tesseract.exe` vào PATH.
4. Trong terminal dự án chạy `python -m pip install -r requirements-plate.txt`.
5. Khởi động lại ứng dụng.

Nếu thiếu model hoặc OCR, ứng dụng ghi rõ trạng thái và bỏ qua biển số; không tự suy đoán.
Biển quá nhỏ, mờ hoặc OCR không đạt ngưỡng thì để trống. Mẫu kiểm tra hiện tại ưu tiên biển dân sự phổ biến, không bao phủ mọi loại biển đặc thù.
