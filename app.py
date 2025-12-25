import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Road Issues Detection",
    layout="centered"
)

IMG_SIZE = (64, 64)

CLASS_NAMES = [
    "Broken Road Sign Issues",
    "Damaged Road issues",
    "Illegal Parking Issues",
    "Littering Garbage on Public Places Issues",
    "Mixed Issues",
    "Pothole Issues",
    "Vandalism Issues"
]

MODEL_PATHS = {
    "CNN Baseline": "saved_models/cnn_baseline.h5",
    "DenseNet-201 (Pretrained)": "saved_models/densenet201.h5",
    "VGG-16 (Pretrained)": "saved_models/vgg16.h5"
}

# ===============================
# LOAD MODEL (CACHE)
# ===============================
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# ===============================
# UI
# ===============================
st.title("Road Issues Detection System")
st.write(
    "Sistem klasifikasi permasalahan infrastruktur jalan "
    "menggunakan CNN dan Transfer Learning."
)

model_choice = st.selectbox(
    "Pilih Model",
    list(MODEL_PATHS.keys())
)

model = load_model(MODEL_PATHS[model_choice])

uploaded_file = st.file_uploader(
    "Upload gambar jalan",
    type=["jpg", "jpeg", "png"]
)

# ===============================
# PREDICTION
# ===============================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Image", use_container_width=True)

    img = image.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Hasil Prediksi:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    st.subheader("📊 Probabilitas Tiap Kelas")
    for cls, prob in zip(CLASS_NAMES, prediction[0]):
        st.write(f"{cls}: {prob*100:.2f}%")
