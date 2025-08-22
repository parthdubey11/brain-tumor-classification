import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image



    
model = load_model("resnet_model.h5")

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