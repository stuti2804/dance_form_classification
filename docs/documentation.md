# Project Documentation

## Overview
This project is a dance pose classification task that uses a convolutional neural network (MobileNetV2) as a feature extractor and applies machine learning models (Random Forest and Support Vector Machine) for classification of different dance styles.

## Steps Breakdown

### 1. Install Required Libraries
Initial cell ensures installation of necessary libraries:
```bash
pip install tensorflow scikit-learn pandas matplotlib
```

### 2. Import Libraries
Imports Python packages for:
- Data handling: `pandas`, `numpy`
- ML models and metrics: `sklearn`
- Deep learning and image preprocessing: `tensorflow.keras`
- Visualization: `matplotlib`

### 3. Data Upload
Uses Google Colab’s `files.upload()` to allow ZIP file upload of the dataset.

### 4. Data Extraction
Unzips the uploaded dance dataset to a local directory.

### 5. Image Preprocessing
- Iterates through class folders to load images.
- Resizes images to 224x224.
- Applies MobileNetV2 preprocessing (`preprocess_input`).
- Assigns class labels and one-hot encodes them.

### 6. Feature Extraction with MobileNetV2
- Loads MobileNetV2 without top layers and applies `GlobalAveragePooling2D`.
- Extracts image features for classification.

### 7. Data Preparation
- Flattens image features.
- Splits into training and testing sets using `train_test_split`.

### 8. Model Training
Trains two ML models:
- **Random Forest**
- **Support Vector Machine (SVM)**

### 9. Model Evaluation
Prints accuracy and classification report using `accuracy_score` and `classification_report`.

### 10. Visualization
Plots some predictions to visualize model performance.

## Notes
- The notebook is designed for Google Colab.
- Data is expected in ZIP format, organized in class-wise directories.