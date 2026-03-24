# CropGuard AI

An AI-powered web application for detecting diseases in crop leaves using deep learning.

## Overview

CropGuard AI is a machine learning project developed as part of an AIML internship. 
The system uses Convolutional Neural Networks (CNN) with transfer learning to classify 
crop leaf images and identify diseases with high accuracy.

## Features

- Detects 12 disease classes across 2 crops
- 96.68% training accuracy
- 92.83% validation accuracy
- Real-time disease detection from uploaded images
- Treatment suggestions for detected diseases
- Simple and intuitive web interface

## Supported Crops and Diseases

### Pepper Bell
- Bacterial Spot
- Healthy

### Tomato
- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Yellow Leaf Curl Virus

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10 | Programming Language |
| TensorFlow 2.15 | Deep Learning Framework |
| Keras | Model Building |
| MobileNetV2 | Transfer Learning Model |
| Streamlit | Web Application Framework |
| NumPy | Numerical Computing |
| Pillow | Image Processing |

## Dataset

- Name: PlantVillage Dataset
- Total Classes: 12
- Source: Kaggle

## Model Architecture

- Base Model: MobileNetV2 (pretrained on ImageNet)
- Additional Layers: GlobalAveragePooling2D, Dense(128), Dropout(0.2), Dense(12)
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Epochs: 10

## Model Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | 96.68% |
| Validation Accuracy | 92.83% |
| Training Loss | 0.0915 |
| Validation Loss | 0.2646 |

## How to Run

1. Clone the repository:
   git clone https://github.com/yourusername/cropguard-ai.git

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

4. Open browser and go to:
   http://localhost:8501

## Project Structure
cropguard-ai/
├── app.py
├── crop_disease_model.h5
├── requirements.txt
└── README.md

## Limitations

- Currently supports only Pepper Bell and Tomato crops
- Model may give incorrect results for non-leaf images
- Requires good quality leaf images for accurate detection

## Future Scope

- Add more crop varieties
- Deploy on mobile application
- Integrate with real-time camera feed
- Add multilingual support for local farmers

## Disclaimer

This application is intended for educational purposes only.
Always consult a local agricultural officer for confirmation.

## Developed By

Reethu
AIML Internship Project 2026

