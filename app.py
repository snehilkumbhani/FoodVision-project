import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt

# --- Load model and class names ---
@st.cache_resource
def load_foodvision_model():
    model = load_model("foodvision_model.keras", safe_mode=False)  # use .keras not .h5
    return model

@st.cache_data
def load_class_names():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    return class_names

model = load_foodvision_model()
class_names = load_class_names()

# --- Streamlit App UI ---
st.set_page_config(page_title="🍔 FoodVision Classifier", page_icon="🍕")

st.title("🍔 FoodVision Classifier")
st.write("Upload a food image and let the model predict its class!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)[0]
    class_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    # Show results
    st.success(f"🍽 Prediction: **{class_names[class_idx]}**")
    st.info(f"✅ Confidence: {round(confidence * 100, 2)}%")

    # --- Show Top 5 Predictions as bar chart ---
    top_indices = predictions.argsort()[-5:][::-1]
    top_classes = [class_names[i] for i in top_indices]
    top_probs = [predictions[i] * 100 for i in top_indices]

    fig, ax = plt.subplots()
    ax.barh(top_classes[::-1], top_probs[::-1])
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top 5 Predictions")
    st.pyplot(fig)
