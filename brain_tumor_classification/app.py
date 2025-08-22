import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os



MODEL_PATH = os.path.join(os.path.dirname(__file__), "resnet_model.h5")

@st.cache_resource  # cache model so it doesn't reload on every run
def load_brain_tumor_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Model file 'resnet_model.h5' not found. Please add it to the project folder.")
        return None
    return load_model(MODEL_PATH)

model = load_brain_tumor_model()

# Class labels
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Streamlit UI
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan image to detect brain tumor type.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]

    st.success(f"Prediction: **{pred_class}**")