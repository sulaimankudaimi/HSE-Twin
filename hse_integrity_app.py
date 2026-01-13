import streamlit as st
import os
import requests
import numpy as np
import plotly.express as px

try:
    import tensorflow as tf
    from ultralytics import YOLO
    import cv2
    HAS_AI = True
except ImportError:
    HAS_AI = False

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="SPC | HSE & Asset Integrity Twin", layout="wide", page_icon="🛡️")

# --- 2. وظيفة تحميل الملف الكبير من Drive ---
def download_large_file(file_id, output):
    url = f'https://drive.google.com/uc?id={file_id}'
    if not os.path.exists(output):
        with st.spinner('📡 Connecting to SPC Cloud to sync AI weights...'):
            response = requests.get(url, stream=True)
            with open(output, 'wb') as f:
                f.write(response.content)
        st.success("✅ Model weights synchronized successfully!")

# معرف الملف من الرابط الذي أرسلته أنت
FILE_ID = '1xghQcu2rDtb6Jp4pvGWs0QUcMJM7NFaE'
AUDIO_MODEL_PATH = 'audio_anomaly_model_v1.h5'

# --- 3. تحميل الموديلات ---
@st.cache_resource
def load_all_brains():
    # تحميل موديل الرؤية
    v_model = YOLO('best.pt')
    
    # تحميل موديل الصوت (بعد التأكد من وجوده)
    download_large_file(FILE_ID, AUDIO_MODEL_PATH)
    a_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    
    return v_model, a_model

vision_m, audio_m = load_all_brains()

# --- 4. تصميم الواجهة الموسعة ---
st.title("🛡️ HSE & Asset Integrity Digital Twin")
st.markdown("Automated Safety Monitoring & Mechanical Diagnostics | **SPC Security Center**")
st.divider()

# صف المراقبة الأول: الرؤية والصوت
col_vision, col_audio = st.columns([2, 1])

with col_vision:
    st.subheader("📹 AI Vision: PPE Compliance")
    # محاكاة كشف YOLO
    st.image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", caption="Live Feed: Monitoring Helmets & Vests", use_container_width=True)
    st.info("AI Logic: Detects (Helmet, No-Helmet, Vest, Worker)")

with col_audio:
    st.subheader("🔊 Asset Acoustic Integrity")
    # عرض "بصمة صوتية" محاكية
    noise = np.random.normal(0, 1, 100)
    fig_audio = px.line(noise, title="Real-time Vibration Signal", template="plotly_dark")
    st.plotly_chart(fig_audio, use_container_width=True)
    st.metric("Vibration Stability", "Normal", delta="-0.02 Hz")

st.divider()

# صف المراقبة الثاني: التوسع (نزاهة الأصول الحرارية)
st.subheader("🌡️ Thermal Integrity & Corrosion Map")
col_t1, col_t2 = st.columns([1, 2])

with col_t1:
    st.write("📝 **Asset Status Summary:**")
    st.write("- **Pipe Segment A-12:** Stable (34°C)")
    st.write("- **Tank 04:** High Oxidation Risk (Pending Inspection)")
    st.error("🔥 Thermal Anomaly Detected in Valve 09")

with col_t2:
    # خريطة حرارية محاكية للأنابيب والخزانات
    thermal_data = np.random.rand(10, 10) * 50
    fig_heat = px.imshow(thermal_data, text_auto=True, color_continuous_scale='RdYlGn_r', title="Surface Temperature Distribution (°C)")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- 5. نظام التنبيهات المركزية ---
st.sidebar.header("🚨 HSE Control Panel")
if st.sidebar.button("Simulate Emergency"):
    st.sidebar.error("EMERGENCY: Personnel detected in danger zone!")
    st.toast("Alert sent to Field Supervisors", icon='📢')

st.sidebar.divider()
st.sidebar.markdown("Designed by **Eng. Solaiman Kudaimi**")
