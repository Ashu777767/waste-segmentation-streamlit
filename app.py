import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Waste Segregation", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/waste_model.h5")

model = load_model()

labels = open("labels.txt").read().splitlines()

st.title("♻️ Waste Segregation System")
st.write("Upload an image of waste to classify")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")
    st.success(f"Predicted Category: **{labels[class_index]}**")
