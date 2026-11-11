r"""CHƯƠNG 6 - NÉN ẢNH (VehicleAI).

Chạy trong thư mục VehicleAI:
    .\.venv\Scripts\python.exe -m streamlit run chuong6_demo.py

Nén và so sánh JPEG (lossy), PNG (lossless) và WebP (lossy).
Không can thiệp vào mô hình nhận diện phương tiện của dự án.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd
import streamlit as st


MAX_PIXELS = 12_000_000


def read_bgr(uploaded_bytes: bytes) -> np.ndarray:
    """Giải mã ảnh về BGR uint8; ảnh có alpha được ghép nền trắng."""
    data = np.frombuffer(uploaded_bytes, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Không đọc được ảnh. Vui lòng chọn JPG, PNG, BMP hoặc WebP hợp lệ.")
    if img.shape[0] * img.shape[1] > MAX_PIXELS:
        raise ValueError("Ảnh quá lớn (hơn 12 triệu pixel). Hãy chọn ảnh nhỏ hơn.")
    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError("Ảnh phải có độ sâu màu 8-bit hoặc 16-bit.")

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3:
        raise ValueError("Định dạng ảnh không được hỗ trợ.")
    if img.shape[2] == 3:
        return img
    if img.shape[2] == 4:
        # Nền trắng giúp JPEG, PNG và WebP đều so sánh cùng ảnh RGB.
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        color = img[:, :, :3].astype(np.float32)
        return np.rint(color * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    raise ValueError("Ảnh không có số lượng kênh màu phù hợp.")


def compress_bgr(image: np.ndarray, fmt: str, value: int) -> tuple[bytes, np.ndarray]:
    """Mã hóa và giải mã lại để có ảnh thật sự sau nén."""
    if fmt == "JPEG":
        ext, params = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, int(value)]
    elif fmt == "PNG":
        ext, params = ".png", [cv2.IMWRITE_PNG_COMPRESSION, int(value)]
    elif fmt == "WebP":
        ext, params = ".webp", [cv2.IMWRITE_WEBP_QUALITY, int(value)]
    else:
        raise ValueError(f"Định dạng không hỗ trợ: {fmt}")

    try:
        ok, encoded = cv2.imencode(ext, image, params)
        if not ok:
            raise RuntimeError("OpenCV không mã hóa được định dạng này.")
        payload = encoded.tobytes()
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Không giải mã được ảnh sau nén.")
    except cv2.error as exc:
        raise RuntimeError(f"Bộ mã hóa {fmt} chưa được hỗ trợ trong OpenCV: {exc}") from exc
    return payload, decoded


def quality_scores(original: np.ndarray, restored: np.ndarray) -> tuple[float, float]:
    """MSE càng nhỏ càng giống gốc; PSNR càng lớn càng ít sai khác."""
    delta = original.astype(np.float32) - restored.astype(np.float32)
    mse = float(np.mean(delta * delta))
    psnr = math.inf if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
    return mse, psnr


def kb(size: int) -> str:
    return f"{size / 1024:,.1f} KB"


def main() -> None:
    st.set_page_config(page_title="Chương 6 - Nén ảnh | VehicleAI", page_icon="📦", layout="wide")
    st.title("CHƯƠNG 6 – PHƯƠNG PHÁP NÉN ẢNH")
    st.write("Thực hành nén ảnh phương tiện bằng JPEG, PNG và WebP, rồi so sánh dung lượng và chất lượng.")
    st.info("JPEG và WebP ở chế độ này nén mất dữ liệu; PNG nén không mất dữ liệu. Chỉ dùng cho ảnh, không cần chạy YOLO.")

    uploaded = st.file_uploader("Bước 1: Tải một ảnh phương tiện", type=["jpg", "jpeg", "png", "bmp", "webp"])
    fmt = st.selectbox("Bước 2: Chọn phương pháp nén", ["JPEG", "PNG", "WebP"])
    if fmt == "PNG":
        setting = st.slider("Mức nén PNG (0: nhanh, 9: nén kỹ hơn)", 0, 9, 6)
        description = "PNG bảo toàn pixel ảnh; mức nén thường chỉ thay đổi dung lượng và thời gian xử lý."
    else:
        setting = st.slider(f"Chất lượng {fmt} (10: nhỏ hơn, 100: giữ chất lượng hơn)", 10, 100, 80, 5)
        description = "Chất lượng càng thấp thường càng giảm dung lượng, nhưng có thể tạo mờ và nhiễu khối."
    st.caption(description)

    if uploaded is None:
        st.write("Chọn một ảnh xe máy, ô tô hoặc xe buýt để bắt đầu.")
        return

    original_bytes = uploaded.getvalue()
    if len(original_bytes) > 25 * 1024 * 1024:
        st.error("Ảnh tải lên quá 25 MB. Vui lòng chọn ảnh nhỏ hơn.")
        return
    try:
        source = read_bgr(original_bytes)
        compressed, restored = compress_bgr(source, fmt, setting)
        mse, psnr = quality_scores(source, restored)
    except (ValueError, RuntimeError) as exc:
        st.error(str(exc))
        return

    h, w = source.shape[:2]
    st.caption(f"Kích thước ảnh: {w} × {h} pixel. So sánh được thực hiện trên cùng dữ liệu pixel đã giải mã.")
    a, b = st.columns(2)
    a.image(cv2.cvtColor(source, cv2.COLOR_BGR2RGB), caption="Ảnh đầu vào (giải mã)", use_container_width=True)
    b.image(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB), caption=f"Ảnh sau nén {fmt}", use_container_width=True)

    change = (1 - len(compressed) / len(original_bytes)) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("File đã tải lên", kb(len(original_bytes)))
    c2.metric("File nén mới", kb(len(compressed)))
    c3.metric("Thay đổi dung lượng", f"{change:+.1f}% tiết kiệm")
    c4.metric("PSNR", "∞ dB" if math.isinf(psnr) else f"{psnr:.2f} dB")
    st.write(f"**MSE:** {mse:.4f} (càng gần 0 càng tốt).")
    ext = {"JPEG": "jpg", "PNG": "png", "WebP": "webp"}[fmt]
    st.download_button(
        f"⬇️ Tải ảnh nén {fmt}", data=compressed,
        file_name=f"vehicleai_chuong6_{fmt.lower()}.{ext}",
        mime={"JPEG": "image/jpeg", "PNG": "image/png", "WebP": "image/webp"}[fmt],
    )

    st.subheader("Bước 3: So sánh nhiều mức nén trên cùng một ảnh")
    candidates = [("JPEG", 95), ("JPEG", 70), ("JPEG", 40), ("PNG", 0), ("PNG", 9), ("WebP", 80)]
    rows = []
    for comp_fmt, value in candidates:
        try:
            buf, result = compress_bgr(source, comp_fmt, value)
            err, peak = quality_scores(source, result)
            rows.append({
                "Phương pháp": f"{comp_fmt} ({'mức ' if comp_fmt == 'PNG' else 'Q='}{value})",
                "Dung lượng (KB)": round(len(buf) / 1024, 1),
                "Tiết kiệm so với file tải lên (%)": round((1 - len(buf) / len(original_bytes)) * 100, 1),
                "PSNR (dB)": "∞" if math.isinf(peak) else f"{peak:.2f}",
                "MSE": round(err, 3),
            })
        except RuntimeError:
            st.warning(f"Không thể so sánh {comp_fmt} trên môi trường này.")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if rows:
        chart_data = pd.DataFrame(rows).set_index("Phương pháp")[["Dung lượng (KB)"]]
        st.bar_chart(chart_data)

    st.warning("Dung lượng file tải lên có thể đã được nén từ trước. Tỷ lệ tiết kiệm có thể âm; không có nghĩa thuật toán bị lỗi. PNG giữ nguyên pixel của ảnh đã giải mã, không phục hồi chi tiết mất từ ảnh JPEG gốc.")
    with st.expander("Gợi ý viết báo cáo Chương 6"):
        st.markdown("""
- **Mục tiêu:** thử nghiệm nén ảnh phương tiện và đánh giá sự đánh đổi dung lượng – chất lượng.
- **JPEG:** nén mất dữ liệu; Q thấp thường tạo file nhỏ và ảnh kém chi tiết hơn.
- **PNG:** nén không mất dữ liệu; ảnh xuất ra giữ nguyên giá trị pixel đầu vào.
- **WebP:** nén mất dữ liệu ở chế độ Q=10–100; có thể cho dung lượng tốt tùy ảnh.
- **Đánh giá:** dùng dung lượng file, MSE và PSNR. MSE = 0 và PSNR = ∞ nghĩa là pixel không đổi.
- **Minh họa:** chụp JPEG Q=95, JPEG Q=40 và PNG mức 9, kèm bảng so sánh.
""")


if __name__ == "__main__":
    main()
