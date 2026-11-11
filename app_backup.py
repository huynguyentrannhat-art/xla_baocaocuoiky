"""Web UI chạy tại máy: python -m streamlit run app.py"""
from pathlib import Path
import tempfile

import cv2
import numpy as np
import streamlit as st

from vehicle_ai.engine import VehicleEngine
from vehicle_ai.drawing import annotate
from vehicle_ai.runner import imwrite_unicode, run_video
from vehicle_ai.reports import rows_for, write_reports

st.set_page_config(page_title='VehicleAI Việt Nam', page_icon='🚙', layout='wide')
st.title('🚘 VehicleAI VN – Nhận diện và phân loại phương tiện')
st.caption('Nhận diện ảnh/video • Màu sắc từng xe • Nhãn tiếng Việt • Phân loại chuyên sâu khi có weights tự huấn luyện')

weights = Path('weights/vehicles_vn_best.pt')
with st.sidebar:
    st.header('Cài đặt nhận diện')
    custom = weights.exists()
    st.write('Mô hình:', 'Nâng cao – Việt Nam' if custom else 'COCO cơ bản – chưa rõ số ghế')
    conf = st.slider('Ngưỡng phát hiện', .15, .85, .38, .05)
    imgsz = st.select_slider('Kích thước ảnh AI', options=[640, 768, 960, 1280], value=768)
    use_seg = st.checkbox('Dùng mặt nạ để đo màu xe', value=True)
    if custom:
        st.info('Đã tìm thấy weights/vehicles_vn_best.pt')
    else:
        st.warning('Chưa có model riêng. Các nhãn 7/16/32 chỗ, container, cứu hỏa… chưa được dự đoán riêng.')
    plate_weights = Path('weights/license_plate_detector.pt')
    if plate_weights.exists():
        st.info('Đã có weights/license_plate_detector.pt. Module sẽ tự phát hiện vùng biển số; cần Tesseract để đọc ký tự.')
    else:
        st.caption('Biển số: chưa có weights/license_plate_detector.pt nên module đang chờ model.')
    st.caption('Màu là ước lượng từ vùng thân xe. Ánh đèn, kính, bóng đổ, decal nhiều màu có thể gây sai lệch.')


@st.cache_resource(show_spinner='Đang tải mô hình AI…')
def engine_for(weights_file: str, threshold: float, size: int, seg: bool):
    return VehicleEngine(weights_file or None, conf=threshold, imgsz=size, use_seg=seg)


mode = st.radio('Nguồn đầu vào', ['Ảnh', 'Chụp bằng webcam', 'Video'], horizontal=True)
file = None
if mode == 'Ảnh':
    file = st.file_uploader('Chọn ảnh phương tiện', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
elif mode == 'Chụp bằng webcam':
    file = st.camera_input('Chụp ảnh từ webcam (trên trình duyệt)')
else:
    file = st.file_uploader('Chọn video', type=['mp4', 'avi', 'mov', 'mkv'])

if file is not None and st.button('🔍 Bắt đầu nhận diện', type='primary'):
    try:
        with st.spinner('Đang xử lý bằng AI…'):
            engine = engine_for(str(weights) if custom else '', conf, imgsz, use_seg)
            if mode != 'Video':
                raw = np.frombuffer(file.getvalue(), dtype=np.uint8)
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if img is None:
                    st.error('Ảnh lỗi hoặc không hỗ trợ định dạng này.')
                    st.stop()
                detections = engine.process(img, tracking=False)
                output = annotate(img, detections, total=len(detections))
                Path('output').mkdir(exist_ok=True)
                saved = Path('output/streamlit_anh_ketqua.png')
                imwrite_unicode(saved, output)
                write_reports(rows_for(detections), Path('output/streamlit_anh_thongke'),
                              {'so_xe_trong_anh': len(detections)})
                st.success(f'Tìm thấy {len(detections)} phương tiện')
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Ảnh ban đầu', use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption='Ảnh nhận diện', use_container_width=True)
                st.dataframe([x.to_dict() for x in detections], use_container_width=True)
                plate_reads = [x for x in detections if x.plate_text]
                if plate_reads:
                    st.success('Đọc được biển số: ' + ', '.join(x.plate_text for x in plate_reads))
                st.download_button('⬇ Tải ảnh kết quả PNG', saved.read_bytes(),
                                   'ketqua_phuongtien.png', mime='image/png')
                st.download_button('⬇ Tải thống kê CSV', Path('output/streamlit_anh_thongke.csv').read_bytes(),
                                   'thongke.csv', mime='text/csv')
            else:
                # Video phải là đường dẫn trên đĩa để OpenCV mở được.
                suffix = Path(file.name).suffix.lower()
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / ('video_input' + suffix)
                    path.write_bytes(file.getvalue())
                    saved, csv_path, json_path, summary = run_video(
                        engine, str(path), Path('output'), show=False, prefix='streamlit_video')
                st.success(f"Đã xử lý {summary.get('so_khung_hinh', 0)} khung hình")
                st.caption('Số lần nhận diện trong video được tính theo khung hình, KHÔNG phải số xe duy nhất.')
                st.json(summary)
                st.video(saved.read_bytes())
                st.download_button('⬇ Tải video đã gắn nhãn', saved.read_bytes(),
                                   'ketqua_phuongtien.mp4', mime='video/mp4')
                st.download_button('⬇ Tải CSV', csv_path.read_bytes(),
                                   'thongke_video.csv', mime='text/csv')
                st.caption('Nếu trình duyệt không phát được video mp4v, hãy tải và mở bằng VLC.')
    except Exception as exc:
        st.error(f'Lỗi: {exc}')
        st.info('Xem START_HERE.md, kiểm tra Internet ở lần tải weights đầu tiên và phiên bản Python.')

st.divider()
st.markdown('**Lưu ý khoa học:** Mô hình cơ bản chỉ biết 5 nhóm COCO: xe đạp, xe máy, ô tô, xe buýt và xe tải. '
            'Những nhãn nhỏ như xe 16 chỗ, xe 32 chỗ, xe cứu hỏa phải có dữ liệu gán nhãn và weights riêng mới nhận diện được.')
