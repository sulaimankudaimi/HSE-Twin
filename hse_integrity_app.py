import streamlit as st
import tensorflow as tf # لملف الـ h5
from ultralytics import YOLO # لملف الـ pt
import numpy as np

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="SPC | HSE & Asset Integrity Twin", layout="wide", page_icon="🛡️")

# --- 2. تحميل "الأدمغة" من الدرايف ---
@st.cache_resource
def load_models():
    # تحميل موديل البصمة الصوتية 
    audio_model = tf.keras.models.load_model('audio_anomaly_model_v1.h5')
    
    # تحميل موديل الرؤية الحاسوبية (YOLO) 
    vision_model = YOLO('best.pt')
    
    return audio_model, vision_model

audio_m, vision_m = load_models()

# --- 3. واجهة المستخدم ---
st.title("🛡️ HSE & Asset Integrity Digital Twin")
st.markdown("Automated Safety Monitoring & Mechanical Diagnostics | **SPC Security Center**")
st.divider()

col_v, col_a = st.columns(2)

with col_v:
    st.subheader("📹 AI Vision Safety Monitor")
    st.info("System linked to: best.pt ")
    # هنا سنضع كود عرض الفيديو لاحقاً
    st.image("https://via.placeholder.com/600x400.png?text=AI+Vision+Scanning...", use_container_width=True)
    st.caption("Status: Monitoring for PPE Compliance (Helmets, Vests)")

with col_a:
    st.subheader("🔊 Mechanical Sound Analysis")
    st.info("System linked to: audio_anomaly_model_v1.h5 ")
    # محاكاة تحليل صوتي
    st.metric("Acoustic Health Score", "98%", delta="Normal Vibration")
    st.success("✅ Pump Integrity: STABLE")

st.divider()
st.warning("⚠️ Critical Alert: Ensure all personnel in Sector 4 are wearing Level 3 PPE.")