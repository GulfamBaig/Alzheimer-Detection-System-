🧠 Alzheimer Detection System
A deep learning-based web application for detecting Alzheimer's disease stages from MRI scans using Convolutional Neural Networks (CNN).

https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white
https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white

📋 Table of Contents
Overview

Features

Model Architecture

Installation

Usage

Project Structure

Technical Details

Disclaimer

License

🎯 Overview
This project implements an AI-powered system that classifies brain MRI scans into four categories of cognitive impairment:

No Impairment - Normal cognitive function

Very Mild Impairment - Early stage cognitive changes

Mild Impairment - Noticeable cognitive decline

Moderate Impairment - Significant cognitive impairment

The system uses a fine-tuned EfficientNetB0 model trained on a comprehensive dataset of brain MRI scans, achieving ~99% accuracy on test data.

✨ Features
🖼️ Easy Image Upload: Support for JPG, JPEG, PNG, and BMP formats

⚡ Real-time Analysis: Fast prediction with confidence scores

📊 Detailed Results: Comprehensive risk assessment and recommendations

🎯 Professional Interface: Medical-grade UI with appropriate styling

🔒 Privacy Focused: Local processing with temporary file handling

📱 Responsive Design: Works on desktop and mobile devices

🏗️ Model Architecture
Base Model
Architecture: EfficientNetB0

Input Size: 224×224×3 (RGB)

Preprocessing: EfficientNet standard preprocessing

Transfer Learning: Fine-tuned on medical imaging data

Custom Layers
python
- EfficientNetB0 Base (without top)
- Flatten Layer
- Dense (128 units, ReLU, L2 regularization)
- Batch Normalization
- Dropout (0.2)
- Dense (128 units, ReLU, L2 regularization)
- Batch Normalization
- Dropout (0.3)
- Output Layer (4 units, Softmax)
Training Details
Dataset: 10,240 training images, 1,279 test images

Classes: 4 (balanced distribution)

Optimizer: Adam (learning rate: 0.001)

Loss Function: Sparse Categorical Crossentropy

Callbacks: Early Stopping, ReduceLROnPlateau

Performance: ~99% test accuracy

🚀 Installation
Prerequisites
Python 3.8 or higher

pip package manager

Step-by-Step Installation
Clone the repository

bash
git clone https://github.com/your-username/alzheimer-detection.git
cd alzheimer-detection
Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Dependencies
The project requires the following Python packages:

txt
streamlit==1.28.0
tensorflow==2.13.0
pillow==10.0.0
numpy==1.24.3
gdown==4.7.1
💻 Usage
Running the Application
Start the Streamlit app

bash
streamlit run app.py
Access the application

Open your web browser

Navigate to http://localhost:8501

The app will automatically load the AI model from Google Drive

How to Use
Upload MRI Scan

Click "Browse files" or drag and drop an MRI image

Supported formats: JPG, JPEG, PNG, BMP

Analyze Image

Click the "Analyze Image" button

Wait for the model to process the image (first time may take longer)

Review Results

View the prediction and confidence level

Read the risk assessment and recommendations

Consult the educational information provided

Example Workflow
python
# The application automatically:
# 1. Downloads model from Google Drive
# 2. Preprocesses the uploaded image
# 3. Runs inference using the CNN model
# 4. Returns classification results with confidence scores
📁 Project Structure
text
alzheimer-detection/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── assets/                         # Additional resources
│   ├── images/                     # Sample images and icons
│   └── docs/                       # Documentation files
│
└── temp/                           # Temporary files (auto-generated)
    └── models/                     # Downloaded model storage
🔧 Technical Details
Model Download
The application automatically downloads the pre-trained model from Google Drive:

Model File: Alzheimer_Detection_model.h5

Drive URL: https://drive.google.com/uc?id=1urOp_O6ocENaQEXkux4ylWgGW3C19cKE

Cache: Model is cached for faster subsequent loads

Image Preprocessing
python
def preprocess_image(image):
    # Convert to RGB
    image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    
    return img_array
Performance Metrics
Overall Accuracy: 99.22%

Precision: 98-100% across classes

Recall: 98-100% across classes

F1-Score: 99% weighted average

⚠️ Disclaimer
IMPORTANT MEDICAL DISCLAIMER

This application is designed for research and educational purposes only. It is not intended for clinical use or medical diagnosis.

❌ Not a Medical Device: This system is not FDA-approved or certified for medical use

❌ Not a Replacement for Professional Diagnosis: Always consult qualified healthcare professionals

❌ Limited Scope: Only analyzes structural MRI scans for research purposes

❌ No Medical Advice: Results should not be used for treatment decisions

The developers and contributors are not liable for any decisions made based on this application's outputs. Always seek professional medical advice for health-related concerns.

📊 Dataset Information
The model was trained on a comprehensive dataset containing:

Total Images: 11,519 MRI scans

Training Set: 10,240 images (balanced across 4 classes)

Test Set: 1,279 images

Classes:

Mild Impairment (2,560 images)

Moderate Impairment (2,560 images)

No Impairment (2,560 images)

Very Mild Impairment (2,560 images)

🔮 Future Enhancements
Multi-modal analysis (combining MRI with clinical data)

Explainable AI features (Grad-CAM heatmaps)

Batch processing capabilities

User authentication and history

Integration with PACS systems

Mobile app development

🤝 Contributing
We welcome contributions from the research community! Please:

Fork the repository

Create a feature branch

Make your changes

Add tests if applicable

Submit a pull request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
Dataset providers and contributors

TensorFlow and Keras communities

Streamlit for the amazing web framework

Medical researchers in the Alzheimer's field

📞 Support
For technical issues or questions:

Create an issue on GitHub

Check the troubleshooting section in the application

Ensure all dependencies are properly installed
