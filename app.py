import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import gdown
import os
import tempfile

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
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f0f2f6;
    }
    .high-risk {
        border-left: 5px solid #ff4b4b;
    }
    .low-risk {
        border-left: 5px solid #00cc96;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

class AlzheimerDetectionSystem:
    def __init__(self):
        self.model = None
        self.class_names = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']
        self.model_loaded = False
        
    @st.cache_resource(show_spinner=False)
    def load_model_from_drive(_self):
        """Load the trained Alzheimer detection model from Google Drive"""
        try:
            # Google Drive file ID from the shareable link
            file_id = "1urOp_O6ocENaQEXkux4ylWgGW3C19cKE"
            url = f"https://drive.google.com/uc?id={file_id}"
            
            # Create temporary directory for model
            temp_dir = tempfile.mkdtemp()
            model_path = os.path.join(temp_dir, "Alzheimer_Detection_model.h5")
            
            # Download model from Google Drive
            with st.spinner("🔄 Downloading model from Google Drive..."):
                gdown.download(url, model_path, quiet=False)
            
            # Load the model
            with st.spinner("🔄 Loading model..."):
                model = tf.keras.models.load_model(model_path)
            
            st.success("✅ Model loaded successfully!")
            return model
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            - Check internet connection
            - Ensure the Google Drive link is accessible
            - Verify the model file exists and is not corrupted
            """)
            return None
    
    def load_model(self):
        """Load model with progress indicators"""
        if not self.model_loaded:
            self.model = self.load_model_from_drive()
            if self.model is not None:
                self.model_loaded = True
    
    def preprocess_image(self, image):
        """Preprocess the image for model prediction"""
        try:
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image to match model input size
            image = image.resize((224, 224))
            
            # Convert to array and preprocess
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Apply the same preprocessing as during training
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
            
            return img_array
        except Exception as e:
            st.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image):
        """Make prediction on the input image"""
        try:
            # Ensure model is loaded
            if not self.model_loaded:
                self.load_model()
            
            if self.model is None:
                st.error("Model failed to load. Please check the connection and try again.")
                return None, None
            
            # Preprocess the image
            processed_image = self.preprocess_image(image)
            
            if processed_image is None:
                return None, None
            
            # Make prediction
            with st.spinner("🔍 Analyzing MRI scan..."):
                prediction = self.model.predict(processed_image, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction)
            
            return self.class_names[predicted_class], confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

def main():
    # Initialize the detection system
    detector = AlzheimerDetectionSystem()
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Alzheimer Detection System</h1>', unsafe_allow_html=True)
    
    # Pre-load model when app starts (optional)
    # detector.load_model()
    
    # Sidebar
    with st.sidebar:
        st.title("About")
        st.info(
            "This AI system helps in detecting Alzheimer's disease from MRI scans. "
            "It classifies brain images into four categories: No Impairment, "
            "Very Mild Impairment, Mild Impairment, and Moderate Impairment."
        )
        
        st.title("Instructions")
        st.markdown("""
        1. Upload a brain MRI scan image
        2. The system will download and load the AI model
        3. View the prediction results
        4. Consult healthcare professionals for diagnosis
        """)
        
        st.title("Model Information")
        st.markdown("""
        - **Architecture**: EfficientNetB0
        - **Input Size**: 224×224 pixels
        - **Classes**: 4 impairment levels
        - **Accuracy**: ~99% on test data
        """)
        
        # Model status
        st.title("System Status")
        if detector.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.warning("⏳ Model will load when needed")
        
        st.title("Disclaimer")
        st.warning(
            "**Important**: This tool is for research and educational purposes only. "
            "Always consult qualified healthcare professionals for medical diagnosis."
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload MRI Scan")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an MRI image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, JPEG, PNG, BMP"
        )
        
        if uploaded_file is not None:
            # Display the uploaded image
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
                
                # File info
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.2f} KB",
                    "Image dimensions": f"{image.size[0]} x {image.size[1]} pixels"
                }
                
                st.write("**File Details:**")
                for key, value in file_details.items():
                    st.write(f"- {key}: {value}")
                
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                st.stop()
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                # Make prediction
                prediction, confidence = detector.predict(image)
                
                if prediction is not None and confidence is not None:
                    # Display results
                    st.subheader("📊 Analysis Results")
                    
                    # Determine risk level and styling
                    if prediction == 'No Impairment':
                        risk_level = "low-risk"
                        risk_text = "Low Risk"
                        risk_color = "#00cc96"
                        emoji = "✅"
                    elif prediction == 'Very Mild Impairment':
                        risk_level = "high-risk"
                        risk_text = "Monitor Closely"
                        risk_color = "#ffa500"
                        emoji = "⚠️"
                    else:
                        risk_level = "high-risk"
                        risk_text = "Requires Attention"
                        risk_color = "#ff4b4b"
                        emoji = "🚨"
                    
                    # Prediction box
                    st.markdown(
                        f"""
                        <div class="prediction-box {risk_level}">
                            <h3>{emoji} Prediction: {prediction}</h3>
                            <p><strong>Confidence Level:</strong> {confidence:.2%}</p>
                            <p><strong>Risk Assessment:</strong> <span style="color:{risk_color}; font-weight:bold">{risk_text}</span></p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # Confidence meter
                    st.write("**Confidence Meter:**")
                    st.progress(float(confidence))
                    
                    # Additional information based on prediction
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.subheader("📝 What this means:")
                    
                    if prediction == "No Impairment":
                        st.success("""
                        The analysis suggests no significant signs of cognitive impairment. 
                        The brain structure appears normal for age-related changes.
                        """)
                    elif prediction == "Very Mild Impairment":
                        st.warning("""
                        Very mild cognitive changes detected. These may be early indicators 
                        that require monitoring and lifestyle considerations.
                        """)
                    elif prediction == "Mild Impairment":
                        st.warning("""
                        Mild cognitive impairment detected. This suggests noticeable changes 
                        that warrant professional medical consultation.
                        """)
                    else:  # Moderate Impairment
                        st.error("""
                        Moderate cognitive impairment detected. These findings indicate 
                        significant changes that require immediate medical attention.
                        """)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommended Actions")
                    recommendations = {
                        "No Impairment": [
                            "Continue with regular health check-ups",
                            "Maintain cognitive activities and healthy lifestyle",
                            "Annual follow-up with healthcare provider"
                        ],
                        "Very Mild Impairment": [
                            "Consult with a neurologist or geriatric specialist",
                            "Schedule regular cognitive assessments (6-12 months)",
                            "Implement brain-healthy diet and physical exercise",
                            "Consider cognitive training exercises"
                        ],
                        "Mild Impairment": [
                            "Seek immediate medical consultation",
                            "Comprehensive neurological evaluation recommended",
                            "Discuss potential treatment options with specialist",
                            "Regular monitoring (3-6 month intervals)"
                        ],
                        "Moderate Impairment": [
                            "Urgent medical attention required",
                            "Referral to Alzheimer's specialist needed",
                            "Develop comprehensive care and support plan",
                            "Consider family counseling and support services"
                        ]
                    }
                    
                    for rec in recommendations.get(prediction, []):
                        st.write(f"• {rec}")
    
    with col2:
        st.subheader("ℹ️ About Alzheimer's Disease")
        
        st.markdown("""
        **Alzheimer's disease** is a progressive neurological disorder that causes 
        brain cells to degenerate and die, leading to continuous decline in 
        thinking, behavioral, and social skills.
        
        ### 🎯 Stages of Cognitive Impairment:
        
        **🔵 No Impairment**
        - Normal cognitive function
        - No significant memory complaints
        - No objective memory deficits
        - Preserved independence in daily activities
        
        **🟢 Very Mild Impairment**
        - Subjective memory complaints
        - Normal objective memory performance
        - Minimal impact on daily activities
        - Preserved independence
        
        **🟡 Mild Impairment**
        - Objective memory impairment
        - Difficulty with complex tasks
        - Some assistance may be needed
        - Noticeable cognitive changes
        
        **🔴 Moderate Impairment**
        - Clear cognitive decline evident
        - Assistance required for daily activities
        - Significant impact on quality of life
        - Professional care often needed
        
        ### 💡 Importance of Early Detection:
        Early detection of Alzheimer's disease can help in:
        - Better treatment planning and intervention
        - Slowing disease progression
        - Improved quality of life management
        - Appropriate care and support planning
        - Family preparation and education
        """)
        
        # Technical information
        st.subheader("🔬 Technical Information")
        st.markdown("""
        **AI Model Details:**
        - **Base Architecture**: EfficientNetB0
        - **Training Data**: 10,240 MRI scans
        - **Validation**: 1,279 test cases
        - **Input Requirements**: 224×224 pixel RGB images
        - **Preprocessing**: EfficientNet standard preprocessing
        
        **Performance Metrics:**
        - Overall Accuracy: ~99%
        - Precision: 98-100% across classes
        - Recall: 98-100% across classes
        """)

if __name__ == "__main__":
    main()
