import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import gdown
import os

# ========================
# Streamlit Page Config
# ========================
st.set_page_config(
    page_title="Alzheimer Detection System",
    page_icon="🧠",
    layout="wide"
)

# ========================
# Custom CSS
# ========================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .mild-impairment { background-color: #fff3cd; border: 2px solid #ffc107; }
    .moderate-impairment { background-color: #f8d7da; border: 2px solid #dc3545; }
    .no-impairment { background-color: #d1edff; border: 2px solid #0d6efd; }
    .very-mild-impairment { background-color: #d4edda; border: 2px solid #28a745; }
    .confidence-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #dc3545, #ffc107, #28a745);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ========================
# Load Model from Google Drive
# ========================
@st.cache_resource
def load_model():
    model_path = "Alzheimer_Detection_model.h5"
    if not os.path.exists(model_path):
        st.info("Downloading model from Google Drive... This may take a minute ⏳")
        url = "https://drive.google.com/uc?id=1urOp_O6ocENaQEXkux4ylWgGW3C19cKE"  # Replace with your file id
        gdown.download(url, model_path, quiet=False)
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ========================
# Preprocess Image
# ========================
def preprocess_image(image):
    image = image.resize((224, 224))  # resize to match model
    img_array = np.array(image)
    if img_array.shape[-1] == 3:  # convert RGB → BGR
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_array = img_array.astype('float32')
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ========================
# Class Names
# ========================
class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# ========================
# Streamlit App
# ========================
def main():
    st.markdown('<div class="main-header">🧠 Alzheimer Detection System</div>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
        This Alzheimer Detection System uses a deep learning model to analyze MRI brain scans 
        and classify them into four categories:
        - **No Impairment**  
        - **Very Mild Impairment**  
        - **Mild Impairment**  
        - **Moderate Impairment**  
    """)

    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload an MRI brain image (JPG/PNG)  
    2. The image will be processed automatically  
    3. View prediction & confidence levels  
    4. Use clear MRI scans for best results  
    """)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload MRI Scan")
        uploaded_file = st.file_uploader("Choose an MRI brain scan image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

            with st.spinner("Analyzing the MRI scan..."):
                model = load_model()
                if model is not None:
                    processed_image = preprocess_image(image)
                    predictions = model.predict(processed_image)
                    predicted_class = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_class]
                    confidence_scores = predictions[0]

                    with col2:
                        st.subheader("Analysis Results")

                        class_colors = {
                            'Mild Impairment': 'mild-impairment',
                            'Moderate Impairment': 'moderate-impairment',
                            'No Impairment': 'no-impairment',
                            'Very Mild Impairment': 'very-mild-impairment'
                        }

                        st.markdown(
                            f'<div class="prediction-box {class_colors[class_names[predicted_class]]}">'
                            f'<h3>Prediction: {class_names[predicted_class]}</h3>'
                            f'<h4>Confidence: {confidence*100:.2f}%</h4>'
                            f'</div>', 
                            unsafe_allow_html=True
                        )

                        st.subheader("Detailed Confidence Scores")
                        for i, (class_name, score) in enumerate(zip(class_names, confidence_scores)):
                            percentage = score * 100
                            st.write(f"**{class_name}**: {percentage:.2f}%")
                            st.markdown(
                                f'<div class="confidence-bar">'
                                f'<div class="confidence-fill" style="width: {percentage}%"></div>'
                                f'</div>', 
                                unsafe_allow_html=True
                            )

                        st.subheader("Interpretation")
                        interpretations = {
                            'No Impairment': "Healthy brain with no significant Alzheimer's signs.",
                            'Very Mild Impairment': "Early subtle changes may indicate beginning cognitive decline.",
                            'Mild Impairment': "Moderate changes suggest mild cognitive impairment.",
                            'Moderate Impairment': "Significant changes indicate moderate Alzheimer's disease. Consult a doctor."
                        }
                        st.info(interpretations[class_names[predicted_class]])

                        st.warning("""
                        **Disclaimer**: This tool is for educational and research purposes only.  
                        Not a substitute for professional medical diagnosis.  
                        Always consult a healthcare professional.  
                        """)

    if uploaded_file is None:
        with col2:
            st.subheader("Sample MRI Scans")
            st.info("Upload a brain MRI image to see the detection system in action.")
            sample_col1, sample_col2 = st.columns(2)
            with sample_col1:
                st.markdown("**Expected Input:**")
                st.image("https://via.placeholder.com/200x200/4CAF50/FFFFFF?text=Brain+MRI", caption="Sample Brain MRI")
            with sample_col2:
                st.markdown("**Supported Formats:**")
                st.write("- JPG / JPEG")
                st.write("- PNG")
                st.write("- Clear MRI scans")

# ========================
# Run App
# ========================
if __name__ == "__main__":
    main()
