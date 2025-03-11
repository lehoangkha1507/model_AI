import streamlit as st
import numpy as np
from tensorflow import keras

# Load mô hình AI và scaler (sử dụng NumPy để lưu và tải scaler thay vì joblib)
model = keras.models.load_model('fs_model_fpt.h5', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
scaler_data = np.load('scaler.npy', allow_pickle=True).item()  # Thay thế joblib bằng numpy

# Hàm chuẩn hóa dữ liệu (thay thế joblib)
def scale_input(data):
    return (data - scaler_data["mean"]) / scaler_data["std"]  # Chuẩn hóa bằng dữ liệu scaler đã lưu

# Hàm dự đoán hệ số an toàn (FS)
def predict_fs(u, luong_mua, thoi_gian_mua, beta):
    input_data = scale_input(np.array([[u, luong_mua, thoi_gian_mua, beta]]))
    fs_value = model.predict(input_data)[0][0]
    
    status = "✅ An toàn" if fs_value >= 1.5 else "⚠️ Cần kiểm tra" if fs_value >= 1.0 else "❌ Nguy hiểm"
    return {"fs": fs_value, "status": status}

# Giao diện Streamlit
st.title("🛰️ Dự đoán Hệ Số An Toàn (FS)")
st.markdown("**Nhập dữ liệu địa chất để mô hình AI tính toán FS**")

# Nhập dữ liệu từ người dùng
u = st.number_input("Áp lực nước (kN/m²)", min_value=0.0, max_value=100.0, value=50.0)
luong_mua = st.number_input("Lượng mưa (mm)", min_value=0.0, max_value=500.0, value=100.0)
thoi_gian_mua = st.number_input("Thời gian mưa (giờ)", min_value=0.0, max_value=48.0, value=5.0)
beta = st.number_input("Độ nghiêng của mặt trượt (°)", min_value=0.0, max_value=90.0, value=30.0)

# Nút bấm dự đoán
if st.button("🔍 Dự đoán Hệ Số An Toàn"):
    result = predict_fs(u, luong_mua, thoi_gian_mua, beta)
    st.success(f"🔮 Hệ số an toàn (FS): {result['fs']:.3f}")
    st.warning(f"🛑 Kết luận: {result['status']}")

# API đơn giản ngay trên Streamlit (các web/app khác có thể gửi request đến URL này)
st.json(result)
