import streamlit as st
import numpy as np
import joblib
from tensorflow import keras
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import threading

# Load mô hình AI và scaler
model = keras.models.load_model('fs_model_fpt.h5', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
scaler = joblib.load('scaler.pkl')

# FastAPI để cung cấp API
api = FastAPI()

# Định nghĩa dữ liệu đầu vào
class InputData(BaseModel):
    u: float
    luong_mua: float
    thoi_gian_mua: float
    beta: float

# Hàm dự đoán hệ số an toàn (FS)
def predict_fs(u, luong_mua, thoi_gian_mua, beta):
    input_data = np.array([[u, luong_mua, thoi_gian_mua, beta]])
    input_data_scaled = scaler.transform(input_data)
    fs_value = model.predict(input_data_scaled)[0][0]

    if fs_value >= 1.5:
        status = "✅ An toàn"
    elif 1.0 <= fs_value < 1.5:
        status = "⚠️ Cần kiểm tra"
    else:
        status = "❌ Nguy hiểm"

    return {"fs": fs_value, "status": status}

# API để lấy dữ liệu FS
@api.post("/api/predict")
async def api_predict(data: InputData):
    result = predict_fs(data.u, data.luong_mua, data.thoi_gian_mua, data.beta)
    return result

# Chạy FastAPI trên luồng riêng
def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8502)

# Tạo luồng để chạy FastAPI song song với Streamlit
api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

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
