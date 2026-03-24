import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="centered"
)

st.markdown("""
    <style>
   .stApp {
        background-image: url("https://tse2.mm.bing.net/th/id/OIP.WxTJG3_OV7cS49I9UlcDVgHaEo?pid=Api&P=0&h=180");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
    }
   .block-container {
        background-color: transparent;
        padding: 2rem;
        margin-top: 2rem;
    }
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        padding-top: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #ffffff;
        text-align: center;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    h2, h3, p, label {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
    }
    .stMetric {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    .stFileUploader {
        background-color: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        padding: 10px;
    }
    .footer {
        text-align: center;
        color: #ffffff;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
        font-size: 13px;
        margin-top: 40px;
    }
    .stButton > button {
        background-color: #1b5e20;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #ffffff;
        color: #1b5e20;
    }
    }
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #1b5e20;
        text-align: center;
        padding-top: 20px;
    }
    .subtitle {
        font-size: 18px;
        color: #388e3c;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton > button {
        background-color: #2e7d32;
        color: white;
        border-radius: 10px;
        border: none;
        font-size: 16px;
        padding: 10px;
    }
    .stButton > button:hover {
        background-color: #1b5e20;
        color: white;
    }
    .stFileUploader {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 10px;
    }
    .stMetric {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #a5d6a7;
    }
    .footer {
        text-align: center;
        color: #388e3c;
        font-size: 13px;
        margin-top: 40px;
    }
    .stAlert {
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌿 CropGuard AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered crop disease detection for Pepper & Tomato plants</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Accuracy", "96.68%")
with col2:
    st.metric("🌱 Crops", "2 Crops")
with col3:
    st.metric("🦠 Diseases", "12 Classes")

st.divider()

@st.cache_resource
def load_my_model():
    return load_model('crop_disease_model.h5')

model = load_my_model()

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_healthy', 'Tomato_Late_blight', 'Tomato_Leaf_Mold',
               'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot', 'Tomato__Tomato_mosaic_virus',
               'Tomato__Tomato_YellowLeaf__Curl_Virus']

treatments = {
    'Pepper__bell___Bacterial_spot': 'Apply copper based fungicide. Remove infected leaves immediately.',
    'Pepper__bell___healthy': 'Plant is healthy! Keep watering regularly.',
    'Tomato_Bacterial_spot': 'Use copper spray. Avoid overhead watering.',
    'Tomato_Early_blight': 'Apply fungicide every 7-10 days. Remove infected leaves.',
    'Tomato_healthy': 'Plant is healthy! No action needed.',
    'Tomato_Late_blight': 'Apply chlorothalonil fungicide immediately.',
    'Tomato_Leaf_Mold': 'Improve air circulation. Apply fungicide.',
    'Tomato_Septoria_leaf_spot': 'Remove infected leaves. Apply mancozeb fungicide.',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Use miticide spray. Keep plants well watered.',
    'Tomato__Target_Spot': 'Apply copper fungicide. Remove infected leaves.',
    'Tomato__Tomato_mosaic_virus': 'No cure. Remove infected plants to prevent spread.',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'No cure. Control whiteflies to prevent spread.'
}

st.subheader("Upload Leaf Image")
st.write("Supported crops: Pepper Bell | Tomato")
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Leaf", use_column_width=True)

    st.divider()

    if st.button("Detect Disease", use_container_width=True):
        with st.spinner("Analyzing leaf..."):
            img = image.resize((224, 224))
            img_array = img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = round(100 * np.max(predictions), 2)

            st.subheader("Results")

            if confidence < 70:
                st.warning("Image not recognized! Please upload a Pepper or Tomato leaf image only.")
            else:
                if 'healthy' in predicted_class.lower():
                    st.success(f"{predicted_class.replace('_', ' ')}")
                else:
                    st.error(f"{predicted_class.replace('_', ' ')}")

                st.metric("Confidence Score", f"{confidence}%")

                st.subheader("Suggested Treatment")
                st.info(treatments[predicted_class])

                st.warning("Please consult a local agricultural officer for confirmation.")

st.markdown('<p class="footer">🌿 Built with TensorFlow & Streamlit | CropGuard AI 2026</p>', unsafe_allow_html=True)