# VehicleAI Việt Nam – Dự án nhận diện/phân loại phương tiện giao thông

**Python 3.11 – VS Code – Ultralytics YOLO26 – OpenCV HSV – Pillow – Streamlit**

Dự án môn **Xử lý ảnh kỹ thuật số**; xử lý ảnh, video và webcam. Không xác định thương hiệu xe. Tên gọi theo cách sử dụng phổ biến tại Việt Nam.

> 🚦 **Trạng thái thực tế (đọc trước):** mã nguồn chạy ở chế độ cơ bản bằng mô hình YOLO26 phân vùng COCO. Chế độ 48 loại chỉ dùng được sau khi bạn thu thập/ghi nhãn bộ ảnh thật và huấn luyện ra `weights/vehicles_vn_best.pt`. Không có file `.pt` 48 loại huấn luyện sẵn trong gói này. Không nên quảng cáo “đạt 99%” hoặc “phân biệt mọi xe” khi chưa đánh giá trên test độc lập.

## 1. Tính năng

- Nhận diện phương tiện trong **nhiều đối tượng cùng ảnh**.
- Nhận diện trên **ảnh tĩnh**, **video**, **camera laptop/USB thời gian thực**.
- **Màu sơn ước lượng theo từng xe** bằng HSV: đỏ, cam, vàng, xanh lá, xanh ngọc, xanh dương, tím, hồng, nâu, trắng, bạc, xám, đen. Nếu chất lượng quá kém/nhiều màu, trả **Không rõ**.
- YOLO segmentation **mặt nạ riêng từng xe** để tránh lấy nền vào phân tích màu (khi model có mask).
- Các nhãn tiếng Việt, điểm dự đoán, vị trí khung xe, ID theo dõi tạm thời trong video.
- **Ảnh kết quả PNG**, **video kết quả MP4**, **CSV UTF-8 BOM mở bằng Excel**, **JSON thống kê**.
- Giao diện Streamlit địa phương + CLI VS Code + công cụ gán nhãn YOLO + script huấn luyện/đánh giá.

## 2. Hai mức nhận diện

### Mức A – chạy được bằng model pretrained công khai

Mô hình `yolo26n-seg.pt` chỉ có **5 nhóm xe** có trong COCO:

| Nhãn mô hình | Nhãn hiển thị (trung thực) |
|---|---|
| `bicycle` | Xe đạp (chưa rõ loại) |
| `motorcycle` | Xe máy/mô tô (chưa rõ loại) |
| `car` | Ô tô con (chưa rõ kiểu và số chỗ) |
| `bus` | Xe buýt/xe khách (chưa rõ số chỗ) |
| `truck` | Xe tải (chưa rõ tải trọng/chức năng) |

**KHÔNG tự đổi `bus` thành “xe khách 16 chỗ”; KHÔNG gọi `truck` là container khi chưa có model học loại này.** Chế độ cơ bản có thể nhầm `bus`/`truck` trong cảnh khó; đó là hạn chế pretrained không chuyên về phương tiện Việt Nam.

### Mức B – danh mục 48 lớp dành cho model tự huấn luyện

Tập lớp tại `data/vehicles_vn.yaml` (thứ tự lớp số 0..47 đúng bằng `vehicle_ai/labels.py`):

**Nhóm xe đạp/xe máy**: Xe đạp, xe đạp điện, xe máy số, xe tay ga, xe máy điện, xe mô tô thể thao, mô tô cruiser, mô tô địa hình/du lịch, xe ba bánh máy, xe ba gác chở hàng, xích lô.

**Ô tô con, xe 4–9 chỗ**: Sedan, hatchback, coupe, mui trần, SUV 5 chỗ, SUV 7 chỗ, MPV 7 chỗ, bán tải, van 9 chỗ.

**Xe khách**: 16 chỗ, 29 chỗ, 32 chỗ, 45 chỗ, xe buýt đô thị, xe khách đường dài, xe đưa đón học sinh.

**Xe đặc biệt**: Cứu thương, cứu hỏa, cảnh sát, cứu hộ giao thông, chở rác, quét đường.

**Xe tải/chuyên dụng**: Tải nhẹ/trung/nặng, xe ben, xe bồn, xe tải đông lạnh, đầu kéo container, đầu kéo rời, xe thùng lửng, trộn bê tông, xe tải gắn cẩu.

**Máy/xe công trình**: Xe nâng, máy xúc, xe lu, máy kéo nông nghiệp.

> Tên nhóm là **mục tiêu huấn luyện**, không phải năng lực của weights mặc định. Loại máy điện, tải trọng và số ghế không thể luôn suy ra từ một ảnh; khi không xác định được phải trả “chưa rõ”. Xe 32 chỗ/29 chỗ có hình dáng gần giống nhau; nhãn đào tạo phải dựa trên thông tin thực tế của phương tiện.

## 3. Quy trình AI

```text
Camera / Ảnh / Video
        |
        v
OpenCV đọc BGR, giữ ảnh màu gốc
        |
        v
YOLO26-SEG pretrained -> bbox và mask COCO
        |
        +---- Nếu có weights VN: YOLO26 detect fine-tuned -> 48 nhãn
        |                                   |
        |<---- Ghép mask COCO theo IoU + nhóm đối tượng
        v
Mẫu pixel màu bên trong vùng thân xe (loại bớt mép, kính tối/lốp)
        |
        v
Chuyển BGR -> HSV -> histogram màu -> màu trội / 'không rõ'
        |
        v
Theo dõi sơ bộ bằng IoU (video) -> Pillow vẽ Unicode
        |
        v
PNG / MP4 / CSV / JSON + Giao diện
```

**Bản chất điểm số:** `do_tin_cay_xe` là điểm đầu ra của detector; `do_tin_cay_mau` thực chất là **tỷ lệ pixel ủng hộ màu trội**, KHÔNG phải xác suất đã được hiệu chuẩn. Khi dùng mask YOLO, mask là dự đoán học sâu nên vẫn có thể chứa kính xe hoặc lỗi phân vùng.

**Cách nhận diện màu:** Không lấy trung bình RGB toàn bbox vì sẽ trộn bầu trời, đường, kính và lốp; thay vào đó lấy pixel ở vùng giữa/thân, ưu tiên mask YOLO phân vùng, chuyển HSV, lọc pixel quá tối khi có vùng sáng phù hợp, thống kê nhóm màu. Ảnh nhiều màu/thiếu pixel -> từ chối xác định.

**Giới hạn của màu:** bóng đổ, phản chiếu, cửa kính, đèn đỏ ban đêm, xe dán hai màu, xe rất nhỏ có thể gây sai. Phân vùng cơ bản vẫn bao gồm vật thể xe nhưng không biết vùng *sơn thân* chính xác; khi cần độ chính xác rất cao, nên huấn luyện riêng **body-part segmentation** và bộ gán nhãn màu trên ảnh chuẩn.

## 4. Ứng dụng nội dung 6 chương PDF đã cung cấp

Các PDF đính kèm là **slide lý thuyết**, không chứa sẵn project Python nhận diện 48 xe. Những điểm được vận dụng:

| Tài liệu | Kỹ thuật vận dụng | Trong mã nguồn |
|---|---|---|
| Chương 1 – Xử lý ảnh kỹ thuật số | Quy trình thu nhận → xử lý → nhận dạng; pixel, kênh RGB | `main.py`, `runner.py`, `engine.py` |
| Chương 2 – Tiền xử lý, thu nhận và biểu diễn ảnh | Camera, số hóa ảnh, không gian màu BGR/RGB | `runner.py`, `colors.py` |
| Chương 3 – Phương pháp phát hiện biên | Cạnh/đặc trưng hình dạng; dùng cho kiểm tra chất lượng sau này | Nền tảng lý thuyết; **không** nói đã dùng Sobel/Canny để phân loại 48 loại xe |
| Chương 4 – Phân vùng ảnh | Vùng quan tâm ROI và vùng đồng nhất theo màu | YOLO instance mask, `colors.py` |
| Chương 5 – Phương pháp chuyển đổi ảnh | Histogram, HSV, lọc nhiễu/chuyển màu | `colors.py` (không CLAHE trên ảnh gốc dùng đo màu để tránh đổi màu thật) |
| Chương 6 – Phương pháp nén ảnh | JPEG/PNG và sự đánh đổi mất mát | `runner.py` lưu PNG ảnh kết quả; MP4 cho video |

Không có thuật toán cổ điển nào từ các chương đơn lẻ có thể tự học sự khác nhau của xe 7/16/32 chỗ mà không có bộ dữ liệu và mô hình phù hợp. Phần học sâu YOLO là **mở rộng kỹ thuật** bên ngoài nội dung các slide.

## 5. Cài đặt và sử dụng

**Đọc kỹ file [`START_HERE.md`](START_HERE.md)**. Tóm tắt trong PowerShell VS Code:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Camera video trực tiếp:

```powershell
.\.venv\Scripts\python.exe main.py --source 0
```

Ảnh:

```powershell
.\.venv\Scripts\python.exe main.py --source "D:\Anh\xe.jpg"
```

Nên chọn Python 3.11 để tránh khác biệt phiên bản; Python 3.12 có thể hỗ trợ tùy wheel của PyTorch/Ultralytics, nhưng hướng dẫn này tập trung Python 3.11.

## 6. Hướng dẫn làm data để **thực sự nâng độ chính xác**

1. Chụp và thu thập các ảnh có quyền sử dụng trong giao thông VN: đa dạng góc chụp trước/sau/ngang, mưa/nắng/tối, từng xe đơn lẻ và cảnh đông.
2. Gán nhãn **tất cả xe nhìn rõ** trong ảnh theo mã lớp nhất quán; kiểm tra **kiểu xe đúng**, không gán tùy tiện xe 32 chỗ dựa vào kích thước.
3. Đảm bảo có ảnh train, val và test riêng, tránh trùng khung/video; cân bằng các nhóm dễ nhầm.
4. Đặt ảnh/nhãn ở `dataset/images/{train,val}` và `dataset/labels/{train,val}`. Có thể dùng **`label_dataset.py`** hoặc CVAT/Label Studio.
5. Chạy `check_dataset.py` kiểm tra lớp, cột và tọa độ.
6. Chạy `train.py` để fine-tune từ `yolo26s.pt` hoặc mẫu lớn hơn. Tùy chỉnh `imgsz`, `epochs`, `batch` theo phần cứng.
7. Dùng `evaluate.py` lấy **mAP50 và mAP50-95**, confusion matrix; kiểm tra bằng video hiện trường chưa từng xuất hiện trong tập train.
8. Sửa nhãn sai, bổ sung ảnh cho cặp lớp nhầm nhiều, rồi train lại. Đừng lấy mAP của COCO làm mAP của model Việt Nam.

### Cấu trúc thư mục

```text
PhanLoaiPhuongTien_VietNam_VSCode/
  main.py                     # CLI ảnh, video, camera
  app.py                      # giao diện trình duyệt Streamlit
  train.py                    # huấn luyện 48 lớp
  evaluate.py                 # đánh giá mAP và confusion matrix
  check_dataset.py            # kiểm tra annotation
  label_dataset.py            # gán nhãn bounding box bằng chuột
  requirements.txt
  START_HERE.md
  README.md
  data/
    vehicles_vn.yaml          # thứ tự 48 lớp
  dataset/
    images/train/ , images/val/
    labels/train/ , labels/val/
  vehicle_ai/
    engine.py                 # model inference & ghép mask
    colors.py                 # HSV màu sắc từng xe
    labels.py                 # tên gọi phương tiện Việt Nam
    tracker.py                # ID đơn giản theo IoU
    drawing.py                # khung tên Unicode
    runner.py                 # nhập/xuất ảnh và video
    reports.py                # CSV + JSON
  weights/
    vehicles_vn_best.pt       # TẠO KHI TRAIN, không đính kèm
  tests/
    test_colors.py
  output/                     # ảnh/video/thống kê khi chạy
  1_CAI_DAT.bat
  2_MO_GIAO_DIEN.bat
  3_MO_CAMERA.bat
```

## 7. Các vấn đề cần nói rõ trong báo cáo

- Nhận diện **loại xe** và ước lượng **màu xe** là 2 bài toán khác nhau; có thể một cái đúng và một cái sai.
- Nhãn 7/16/32 chỗ trên hình ảnh là phân loại **kiểu hình học được huấn luyện**, không phải đọc giấy đăng kiểm hoặc đếm ghế chính xác.
- Máy điện và máy xăng có thể rất giống nhau về ngoại thất; không nên khẳng định loại động cơ khi ảnh thiếu dấu hiệu.
- Nhận biết xe cứu thương/cứu hỏa phụ thuộc họa tiết, thiết bị và điều kiện ảnh chụp; không lấy màu đỏ/trắng duy nhất để phân loại.
- Xe đầu kéo/container cần nhìn đủ cấu trúc sơ-mi rơ-moóc; xe có cab đầu kéo không đồng nghĩa luôn có container.
- Đối tượng bị che khuất, độ phân giải thấp, mờ/đêm có thể **không nhận diện** hoặc **không rõ màu**.
- Thống kê video là số lần xuất hiện trên các khung hình. ID dựa IoU chỉ là **theo dõi đơn giản**, không chứng minh đếm chính xác số xe đi qua.
- FPS tùy GPU/CPU, độ phân giải và số xe; chưa đo benchmark trên máy người dùng.
- Quá trình cài và tải trọng số yêu cầu Internet; bản ZIP không chứa trọng số COCO hoặc dataset bản quyền.

## 8. Kiểm thử mã

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Unit test xác minh màu tổng hợp, map danh mục và IoU. Để xác minh AI cần ảnh thực tế có ground truth và đánh giá riêng.

## 9. Tài liệu kỹ thuật bổ sung

- Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26/
- Object Detection & Custom Dataset: https://docs.ultralytics.com/tasks/detect/
- Ultralytics Training: https://docs.ultralytics.com/modes/train/
- Python Windows: https://www.python.org/downloads/
- VS Code Python: https://code.visualstudio.com/docs/languages/python
- OpenCV: https://docs.opencv.org/

**Giấy phép:** Ultralytics có chính sách AGPL-3.0 và giấy phép Enterprise. Với dự án trường học thường chỉ cần tuân thủ giấy phép nguồn mở; nếu triển khai thương mại/đóng mã nguồn, cần kiểm tra điều kiện sử dụng hiện hành tại https://www.ultralytics.com/license.


## 12. Module biển số đã bổ sung

`vehicle_ai/plate.py` là module tùy chọn, tích hợp vào pipeline ảnh/video và CSV/JSON.
Để đọc được biển số thật, người dùng phải cung cấp `weights/license_plate_detector.pt`
là model YOLO đã huấn luyện phát hiện vùng biển số, đồng thời cài Tesseract OCR và
`pip install -r requirements-plate.txt`. Gói ZIP không chứa model biển số hoặc dữ liệu
huấn luyện biển số. Nếu vùng biển quá nhỏ/mờ, OCR không đạt ngưỡng hoặc không đúng
mẫu biển dân sự phổ biến, hệ thống bỏ trống thay vì đoán.

## 13. Những giới hạn chưa thể khắc phục chỉ bằng sửa code

- Dataset train/val trong ZIP đang trống; cần ảnh và nhãn thực tế trước khi huấn luyện 48 lớp.
- `yolo26n-seg.pt` là model COCO cơ bản, không phải model 48 lớp Việt Nam.
- Muốn phân biệt số chỗ, tải trọng, loại xe cứu hỏa/container phải có dữ liệu gán nhãn được xác thực.
- Màu HSV là ước lượng màu trội trong vùng thân xe, không phải nhận dạng màu sơn được hiệu chuẩn.
- Tracker IoU là theo dõi đơn giản; có thể đổi ID khi xe che khuất hoặc di chuyển nhanh.

## 4. Hai module mô hình riêng

### Module A – Phân loại phương tiện Việt Nam 48 lớp

- Danh mục 48 lớp: `data/vehicles_vn.yaml` và `vehicle_ai/labels.py`.
- Module nạp/kiểm tra weights: `vehicle_ai/vehicle_classifier.py`.
- Huấn luyện: `python train.py --epochs 100 --imgsz 960 --batch 8 --device 0`.
- Kết quả: `weights/vehicles_vn_best.pt`.
- Chỉ dự đoán được các lớp có ảnh đã gán nhãn đủ tốt. YAML 48 lớp chỉ là danh mục mục tiêu, không có nghĩa model pretrained mặc định đã biết 48 lớp.

### Module B – Phát hiện và đọc biển số Việt Nam

- Detector riêng: `vehicle_ai/plate.py`.
- Engine tự gọi detector biển số sau khi phát hiện xe; biển số đọc được sẽ hiện trên khung ảnh và trong CSV/JSON.
- Đặt model YOLO biển số tại `weights/license_plate_detector.pt`.
- Tạo `data/plate.yaml` từ `data/plate.yaml.example`, chỉnh đường dẫn và tên lớp đúng dataset của bạn.
- Huấn luyện detector riêng: `python train_plate.py --data data/plate.yaml --epochs 100 --imgsz 960 --batch 8 --device 0`.
- Cài OCR Python: `pip install -r requirements-plate.txt`.
- Windows cần cài ứng dụng Tesseract OCR riêng. Nếu chưa tự nhận diện đường dẫn, đặt biến môi trường `TESSERACT_CMD` trỏ đến `tesseract.exe`.

**Hai model độc lập:** `vehicles_vn_best.pt` nhận diện phương tiện; `license_plate_detector.pt` tìm vùng biển số. Tesseract chỉ đọc ký tự sau khi cắt vùng biển. Ảnh biển số nhỏ, mờ, lóa hoặc nghiêng quá mức sẽ được bỏ qua thay vì tự đoán.
\n\n## Dataset đã tích hợp từ người dùng\n\nĐọc `DATASET_README_VI.md`. Gói hiện chứa dataset biển số YOLO và dataset phân loại phương tiện 4 lớp. Dataset phương tiện không có bbox nên chưa thể train detector 48 lớp.\n