# Chest X-Ray Explainable AI (XAI) Diagnostic System

A comprehensive deep learning web application built with Flask for diagnosing Cardiomegaly from chest X-rays. This project goes beyond simple classification by integrating a robust suite of **Explainable AI (XAI)** techniques. These methods visualize and interpret the model's decision-making process, ensuring clinical trust and transparency.

## Features

*   **Deep Learning Diagnosis**: Utilizes a custom Convolutional Neural Network (`heart_xray_cnn.h5`) to predict whether an X-ray is Normal or exhibits Cardiomegaly, providing confidence scores.
*   **Comprehensive XAI Suite**: Implements multiple visualization and interpretability techniques to highlight the regions the model focuses on:
    *   Saliency Maps
    *   Grad-CAM, Smooth Grad-CAM, and Guided Grad-CAM
    *   LIME (Local Interpretable Model-Agnostic Explanations)
    *   SHAP (SHapley Additive exPlanations)
    *   LRP (Layer-wise Relevance Propagation)
    *   Counterfactual Explanations
    *   Combined Heatmaps (Ensemble of Smooth, Guided, and LRP)
*   **Natural Language Explanations**: Integrates with LLMs to translate technical XAI outputs into human-readable text. It offers both "Expert" (clinical) and "Naive" (patient-friendly) explanation levels.
*   **Trust & Verification Dashboards**:
    *   **Model Verification (`/verify`)**: A batch processing dashboard to test the model on multiple images and evaluate its focus on the anatomical heart region using bounding-box overlap metrics.
    *   **XAI Validation (`/validate_xai`)**: A dedicated interface to benchmark and compare the accuracy and localization capabilities of different XAI methods.
*   **Modern UI**: A responsive, dark-themed user interface designed for ease of use in clinical or research settings.

## Project Structure

```text
.
├── app.py                      # Main Flask application and API endpoints
├── heart_xray_cnn.h5           # Pre-trained Keras CNN model
├── requirements.txt            # Python dependencies
├── natural_explanation.py      # LLM integration for natural language explanations
├── templates/                  # HTML templates
│   ├── index.html              # Main diagnostic interface
│   ├── verify.html             # Batch verification dashboard
│   └── validate_xai.html       # XAI technique comparison dashboard
├── static/                     # CSS, JavaScript, and static assets
│   └── style.css
├── test/ & train/              # Sample image directories for testing/training
└── ...                         # Standalone scripts for individual XAI methods
```

## Setup & Installation

1.  **Clone the Repository** (or navigate to the project folder):
    ```bash
    cd "xAI proj"
    ```

2.  **Create a Virtual Environment** (recommended):
    ```bash
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**:
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000/`.

## Usage Guide

1.  **Main Diagnosis (`/`)**: 
    *   Upload a chest X-ray image.
    *   View the prediction (Normal/Cardiomegaly) and explore different tabs to see various XAI heatmaps overlaying the original image.
    *   Generate a natural language explanation of the diagnosis.
2.  **Batch Verification (`/verify`)**: 
    *   Upload multiple X-rays to verify how often the model correctly localizes its attention on the heart region.
  
## Technologies Used

*   **Backend**: Python, Flask
*   **Machine Learning**: TensorFlow / Keras, OpenCV, NumPy
*   **XAI Libraries**: `shap`, `lime`, custom gradient tape implementations
*   **Frontend**: HTML5, Vanilla CSS (Dark Theme), Vanilla JavaScript

## License

This project is intended for educational purpose only
