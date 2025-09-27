import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

# Set page configuration
st.set_page_config(
    page_title="Alzheimer Detection System",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS for better styling
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

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('Model/Alzheimer_Detection_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the image
def preprocess_image(image):
    # Resize image to match model input size (224x224)
    image = image.resize((224, 224))
    # Convert to numpy array
    img_array = np.array(image)
    # Convert RGB to BGR (if needed)
    if img_array.shape[-1] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    # Normalize pixel values
    img_array = img_array.astype('float32')
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Class names mapping
class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# Main app
def main():
    st.markdown('<div class="main-header">🧠 Alzheimer Detection System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This Alzheimer Detection System uses a deep learning model to analyze MRI brain scans 
        and classify them into four categories:
        
        - **No Impairment**: Healthy brain scan
        - **Very Mild Impairment**: Early signs of cognitive decline
        - **Mild Impairment**: Moderate cognitive impairment
        - **Moderate Impairment**: Advanced cognitive impairment
        
        Upload an MRI brain scan image to get started.
        """
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload a brain MRI image (JPEG, PNG, or JPG format)
    2. The image will be automatically processed
    3. View the prediction results and confidence levels
    4. For best results, use clear, high-quality MRI scans
    """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload MRI Scan")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an MRI brain scan image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear MRI scan of the brain for analysis"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            
            # Preprocess and make prediction
            with st.spinner("Analyzing the MRI scan..."):
                try:
                    # Load model
                    model = load_model()
                    
                    if model is not None:
                        # Preprocess image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        predictions = model.predict(processed_image)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class]
                        
                        # Get all confidence scores
                        confidence_scores = predictions[0]
                        
                        # Display results
                        with col2:
                            st.subheader("Analysis Results")
                            
                            # Prediction box with color coding
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
                            
                            # Confidence scores for all classes
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
                            
                            # Interpretation
                            st.subheader("Interpretation")
                            interpretations = {
                                'No Impairment': "The MRI scan appears to show a healthy brain with no significant signs of Alzheimer's disease.",
                                'Very Mild Impairment': "The scan shows early, subtle changes that may indicate the beginning stages of cognitive decline.",
                                'Mild Impairment': "Moderate changes are visible, suggesting mild cognitive impairment that should be monitored closely.",
                                'Moderate Impairment': "Significant changes indicate moderate Alzheimer's disease. Consultation with a healthcare professional is recommended."
                            }
                            
                            st.info(interpretations[class_names[predicted_class]])
                            
                            # Important disclaimer
                            st.warning("""
                            **Important Disclaimer**: 
                            This tool is for educational and research purposes only. 
                            It should not be used as a substitute for professional medical diagnosis. 
                            Always consult with qualified healthcare professionals for medical advice.
                            """)
                            
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.info("Please try uploading a different image or check if the image is a valid brain MRI scan.")
    
    # Display sample images if no file is uploaded
    if uploaded_file is None:
        with col2:
            st.subheader("Sample MRI Scans")
            st.info("Upload a brain MRI image to see the detection system in action.")
            
            # Create sample layout
            sample_col1, sample_col2 = st.columns(2)
            
            with sample_col1:
                st.markdown("**Expected Input:**")
                st.image("https://via.placeholder.com/200x200/4CAF50/FFFFFF?text=Brain+MRI", 
                        caption="Sample Brain MRI Scan", use_column_width=True)
            
            with sample_col2:
                st.markdown("**Supported Formats:**")
                st.write("- JPEG (.jpg, .jpeg)")
                st.write("- PNG (.png)")
                st.write("- High-quality MRI scans")
                st.write("- Clear, focused brain images")

# Run the app
if __name__ == "__main__":
    main()
