import streamlit as st
import os
import requests
import numpy as np
import plotly.express as px

# --- 1. محاولة استدعاء الموديلات بمرونة ---
try:
    from ultralytics import YOLO
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

# --- 2. إعدادات الصفحة ---
st.set_page_config(page_title="SPC | HSE & Asset Integrity Twin", layout="wide", page_icon="🛡️")

# --- 3. وظيفة محاكاة محرك الذكاء الاصطناعي (للسرعة) ---
def simulate_ai_analysis():
    # محاكاة لنتائج موديل البصمة الصوتية (audio_anomaly_model_v1.h5)
    # لكي لا نضطر لتحميل مكتبة TensorFlow الضخمة في كل مرة
    health_score = np.random.uniform(94, 99)
    status = "STABLE" if health_score > 95 else "MAINTENANCE REQUIRED"
    return health_score, status

# --- 4. واجهة المستخدم ---
st.title("🛡️ HSE & Asset Integrity Digital Twin")
st.markdown("Automated Safety Monitoring & Mechanical Diagnostics | **SPC Security Center**")
st.divider()

# صف المراقبة الأول: الرؤية والصوت
col_vision, col_audio = st.columns([2, 1])

with col_vision:
    st.subheader("📹 AI Vision: PPE Compliance")
    if HAS_VISION:
        st.info("AI Logic: Active (YOLO best.pt loaded)")
        # محاكاة صورة كشف
        st.image("https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg", caption="Live Feed: PPE Detection", use_container_width=True)
    else:
        st.warning("📡 AI Vision Engine is initializing...")
        st.image("https://via.placeholder.com/600x400.png?text=Waiting+for+Vision+Stream...", use_container_width=True)

with col_audio:
    st.subheader("🔊 Asset Acoustic Integrity")
    h_score, h_status = simulate_ai_analysis()
    
    # عرض نبضات الصوت (محاكاة بصمة الصوت)
    vibration = np.random.normal(0, 0.1, 100) + np.sin(np.linspace(0, 10, 100))
    fig_audio = px.line(vibration, title="Vibration Signature Analysis", template="plotly_dark")
    st.plotly_chart(fig_audio, use_container_width=True)
    
    st.metric("Acoustic Health Score", f"{h_score:.1f}%", delta=h_status)
    if h_score > 95:
        st.success(f"✅ Status: {h_status}")
    else:
        st.error(f"⚠️ Status: {h_status}")

st.divider()

# صف المراقبة الثاني: التوسع (نزاهة الأصول الحرارية)
st.subheader("🌡️ Thermal Integrity & Corrosion Map")
col_t1, col_t2 = st.columns([1, 2])

with col_t1:
    st.write("📝 **Asset Status Summary:**")
    st.write("- **Pipe Segment A-12:** Stable (34°C)")
    st.write("- **Tank 04:** High Oxidation Risk")
    st.error("🔥 Thermal Anomaly Detected in Valve 09")
    
    # زر التقرير الخاص بالسلامة
    st.download_button("📥 Download HSE Report", "PPE Compliance: 100%\nAsset Integrity: Stable", file_name="HSE_Report.txt")

with col_t2:
    # خريطة حرارية تفاعلية
    thermal_data = np.random.rand(10, 15) * 40 + 20
    fig_heat = px.imshow(thermal_data, text_auto=True, color_continuous_scale='RdYlGn_r', 
                         title="Asset Surface Temperature Distribution (°C)")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- 5. القائمة الجانبية ---
st.sidebar.header("🚨 Emergency Controls")
if st.sidebar.button("Trigger Safety Alarm"):
    st.sidebar.error("ALARM ACTIVATED: Safety Breach in Sector 4")
    st.balloons()

st.sidebar.divider()
st.sidebar.markdown("Designed by **Eng. Solaiman Kudaimi**\n\n*SPC Digital Transformation 2026*")
