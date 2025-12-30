import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# Page style
# Page style - Skin Background (updated for latest Streamlit)
# Gorgeous UI Theme ✨
st.markdown("""
    <style>

    /* FULL APP PURPLE GRADIENT BACKGROUND */
    body, .stApp {
        background: linear-gradient(135deg, #E6D4FF 0%, #C9A7FF 45%, #A36AFF 100%);
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
        color: #2A003D;
    }

    /* PAGE TITLE */
    h1 {
        text-align: center !important;
        color: #3B0057 !important;
        font-size: 2.6rem !important;
        font-weight: 800 !important;
        text-shadow: 1px 1px 3px rgba(255,255,255,0.5);
    }

    /* Subtitle + Description */
    p, span, label {
        text-align: center !important;
        color: #30004A !important;
        font-size: 1.1rem !important;
        font-weight: 500;
    }

    /* UPLOAD BOX */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.25);
        padding: 20px;
        border-radius: 12px;
        border: 2px dashed #7A22D1;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        backdrop-filter: blur(10px);
        transition: 0.3s ease-in-out;
    }
    .stFileUploader:hover {
        border-color: #A35BFF;
        transform: scale(1.03);
    }

    /* IMAGE PREVIEW */
    img {
        border-radius: 20px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
        margin: auto;
        display: block;
    }

    /* BUTTON STYLING */
    .stButton button {
        background-color: #7A22D1 !important;
        color: #FFFFFF !important;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        border: none;
        cursor: pointer;
        transition: 0.2s ease-in-out;
    }
    .stButton button:hover {
        background-color: #9C47FF !important;
        transform: scale(1.07);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    /* PREDICTION BOX (Glass Effect Purple) */
    .prediction-box {
        background: rgba(255, 255, 255, 0.3);
        border: 2px solid rgba(122, 34, 209, 0.3);
        border-radius: 18px;
        padding: 15px;
        margin: 12px auto;
        width: 75%;
        text-align: center;
        box-shadow: 0 4px 25px rgba(0,0,0,0.2);
        backdrop-filter: blur(12px);
        font-size: 1.15rem;
        font-weight: 600;
    }

    /* HEADINGS */
    h2, h3 {
        text-align: center !important;
        color: #4A006E !important;
        font-weight: 700 !important;
    }

    </style>
""", unsafe_allow_html=True)



# Load trained model
model_path = "C:/Users/Samhi/Desktop/dogbreed/dog_breed_model.h5"
model = tf.keras.models.load_model(model_path)

# Load label names
labels = np.load("C:/Users/Samhi/Desktop/dogbreed/allDoglables.npy")
categories = np.unique(labels)

# Preprocess function
def prepare_image(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    img_result = np.expand_dims(resized, axis=0)
    img_result = preprocess_input(img_result)
    return img_result, img_rgb

st.title("🐶 Dog Breed Detection System")
st.write("Upload a dog image to identify its breed!")

uploaded_file = st.file_uploader("Choose a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    processed_img, display_img = prepare_image(img)

    st.image(display_img, channels="RGB", caption="Uploaded Image")

    if st.button("Predict Breed"):
        predictions = model.predict(processed_img)[0]

        # Top 3 predictions
        top3_idx = predictions.argsort()[-3:][::-1]
        top3_scores = predictions[top3_idx]

        st.subheader("🐶 Top 3 Predictions:")

        for i in range(3):
            breed = categories[top3_idx[i]]
            confidence = top3_scores[i] * 100
            
            # Prediction Box with Progress Bar
            st.markdown(
                f"<div class='prediction-box'><b>{i+1}. {breed}</b><br>"
                f"<progress value='{confidence:.2f}' max='100'></progress> "
                f"{confidence:.2f}%</div>",
                unsafe_allow_html=True
            )

        # Create downloadable text report
        report_text = "Dog Breed Prediction Report\n"
        report_text += "---------------------------\n"
        for i, idx in enumerate(top3_idx):
            report_text += f"{i+1}. {categories[idx]}: {predictions[idx]*100:.2f}%\n"

        st.download_button(
            label="📥 Download Prediction Report",
            data=report_text,
            file_name="dog_breed_prediction.txt",
            mime="text/plain"
        )
