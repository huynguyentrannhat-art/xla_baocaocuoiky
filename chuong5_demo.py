r"""Minh hoạ Chương 5: chuyển đổi ảnh, histogram và lọc nhiễu.
Chạy: .\.venv\Scripts\python.exe -m streamlit run chuong5_demo.py
"""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Chương 5 - Chuyển đổi ảnh", page_icon="🖼️", layout="wide")
st.title("CHƯƠNG 5 – PHƯƠNG PHÁP CHUYỂN ĐỔI ẢNH")
st.write("Thực hành trên ảnh phương tiện: ảnh xám, HSV, cân bằng histogram, gamma và lọc nhiễu.")
st.info("Đây là bản minh họa riêng; ảnh đã tăng cường không thay thế ảnh gốc khi ước lượng màu sơn xe.")

file = st.file_uploader("Bước 1: Chọn ảnh (.jpg, .png, .webp)", type=["jpg", "jpeg", "png", "webp", "bmp"])
methods = [
    "Ảnh xám (Grayscale)", "Kênh H trong HSV", "Kênh S trong HSV",
    "Kênh V trong HSV", "Cân bằng histogram ảnh xám",
    "CLAHE (tăng tương phản)", "Gamma (điều chỉnh sáng)",
    "Lọc Gaussian (giảm nhiễu)", "Lọc Median (giảm nhiễu muối tiêu)",
    "Ảnh âm bản (Negative)",
]
method = st.selectbox("Bước 2: Chọn phép chuyển đổi", methods)
gamma = st.slider("Giá trị gamma (dùng cho chức năng Gamma)", .3, 3.0, 0.7, .1)
ksize = st.slider("Kích thước kernel lọc", 3, 11, 5, 2)

if file is not None:
    raw=np.frombuffer(file.getvalue(), dtype=np.uint8)
    bgr=cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Không đọc được ảnh; hãy chọn ảnh khác.")
        st.stop()
    if max(bgr.shape[:2])>1800:
        scale=1800/max(bgr.shape[:2])
        bgr=cv2.resize(bgr,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    notes=""
    if method==methods[0]:
        output=gray
        notes="Chuyển BGR thành ảnh 1 kênh mức xám, giảm thông tin màu nhưng giữ độ sáng tương đối."
    elif method==methods[1]:
        output=np.uint8(hsv[:,:,0]*255/179)
        notes="H mô tả sắc độ (Hue); được chuẩn hoá để xem dưới dạng ảnh xám minh họa."
    elif method==methods[2]:
        output=hsv[:,:,1]
        notes="S mô tả độ bão hòa màu (Saturation)."
    elif method==methods[3]:
        output=hsv[:,:,2]
        notes="V mô tả độ sáng (Value)."
    elif method==methods[4]:
        output=cv2.equalizeHist(gray)
        notes="Cân bằng histogram để phân bố lại mức xám, có thể làm rõ tương phản."
    elif method==methods[5]:
        lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        enhanced=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(l)
        output=cv2.cvtColor(cv2.merge((enhanced,a,b)),cv2.COLOR_LAB2BGR)
        notes="CLAHE tăng tương phản cục bộ trên kênh độ sáng LAB; có thể đổi sắc độ hiển thị."
    elif method==methods[6]:
        table=np.array([round(((x/255.)**gamma)*255.) for x in range(256)],dtype=np.uint8)
        output=cv2.LUT(bgr,table)
        notes="s = 255 × (r/255)^gamma; gamma < 1 làm sáng, gamma > 1 làm tối."
    elif method==methods[7]:
        output=cv2.GaussianBlur(bgr,(ksize,ksize),0)
        notes="Lọc Gaussian làm mượt nhiễu nhưng có thể mờ biên."
    elif method==methods[8]:
        output=cv2.medianBlur(bgr,ksize)
        notes="Lọc trung vị thường hữu ích đối với nhiễu muối tiêu."
    else:
        output=255-bgr
        notes="Âm bản đảo mức sáng 255-r, minh hoạ biến đổi điểm."

    left,right=st.columns(2)
    left.image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),caption="Ảnh gốc (BGR → RGB)",use_container_width=True)
    if output.ndim==2:
        right.image(output,caption="Ảnh sau chuyển đổi (1 kênh)",clamp=True,use_container_width=True)
        result_gray=output
    else:
        right.image(cv2.cvtColor(output,cv2.COLOR_BGR2RGB),caption="Ảnh sau chuyển đổi",use_container_width=True)
        result_gray=cv2.cvtColor(output,cv2.COLOR_BGR2GRAY)
    st.write("**Giải thích:**",notes)
    st.subheader("Bước 3: So sánh histogram mức xám")
    def hist(channel):
        return cv2.calcHist([channel],[0],None,[256],[0,256]).flatten()
    df=pd.DataFrame({"Gốc":hist(gray),"Sau xử lý":hist(result_gray)},index=list(range(256)))
    st.line_chart(df)
    success,encoded=cv2.imencode('.png',output)
    if success:
        st.download_button("Tải ảnh sau xử lý (.png)",encoded.tobytes(),"chuong5_ketqua.png",mime="image/png")
    st.caption("Chụp màn hình cặp ảnh và đồ thị để minh họa Chương 5 trong báo cáo.")
else:
    st.caption("Hãy chọn một ảnh có xe để bắt đầu thực hành.")
