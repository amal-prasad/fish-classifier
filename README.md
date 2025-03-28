# Multiclass Fish Image Classification

## Project Aim and Intention

The goal of this project is to develop a robust deep learning solution that can accurately classify images of fish into multiple categories. This involves:

- **Enhanced Accuracy:** Determining the best model architecture to improve the classification performance.
- **Deployment Ready:** Creating a user-friendly web application that can perform real-time predictions.
- **Model Comparison:** Evaluating and comparing different models to select the most suitable approach for fish image classification.

## Problem Statement

This project focuses on classifying fish images into multiple categories using deep learning models. The task involves:

- **Training a CNN from Scratch:** Building a custom convolutional neural network that learns features directly from the fish images.
- **Leveraging Transfer Learning:** Utilizing pre-trained models to enhance performance by fine-tuning them on the fish dataset.
- **Model Saving and Deployment:** Saving the trained models for later use and deploying a Streamlit application that enables users to upload images and receive predictions in real time.

## How the Project Was Carried Out

### 1. Data Preprocessing
- **Image Rescaling:** The images were rescaled to a normalized range [0, 1] for efficient model training.
- **Data Augmentation:** Techniques such as rotation, zoom, and flipping were applied to enhance the diversity of the dataset and improve model robustness.

### 2. Model Development
- **Custom CNN Model:** A convolutional neural network was built from scratch to learn features specific to the fish images.
- **Transfer Learning:** Five pre-trained models (e.g., VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0) were fine-tuned on the fish dataset to leverage their robust feature extraction capabilities.

### 3. Model Evaluation
- **Metrics Comparison:** The performance of each model was evaluated using metrics such as accuracy, precision, recall, and F1-score.
- **Visualization:** Training history, including accuracy and loss curves, was visualized to assess model performance and convergence.

### 4. Deployment
- **Streamlit Application:** A user-friendly web application was built using Streamlit. This app allows users to:
  - Upload fish images.
  - Receive real-time predictions along with model confidence scores.
- **Model Persistence:** The best-performing model was saved in a suitable format (e.g., `.h5` or `.pkl`) for future use.

## Business Use Cases

- **Enhanced Accuracy:** By comparing multiple models, the project identifies the most effective architecture for fish classification.
- **Deployment Ready:** The creation of an interactive web application provides an easy-to-use interface for end-users.
- **Model Comparison:** Detailed evaluation metrics enable a thorough comparison across different models, ensuring the best approach is selected for practical deployment.

---

*This README provides an overview of the project's aims, the methodology employed, and the business cases driving the development process.*
