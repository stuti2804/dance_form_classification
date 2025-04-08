# Dance Pose Classification

This project performs image classification on dance styles using deep learning for feature extraction and traditional machine learning classifiers for prediction.

## Features
- Uses **MobileNetV2** for image feature extraction
- Trains **Random Forest** and **SVM** classifiers
- Evaluates performance using classification metrics
- Visualizes predictions on test data

## How to Run

1. Clone or open the notebook in **Google Colab**
2. Upload a zipped dataset of dance images (organized into folders by class)
3. Run all cells sequentially

## Folder Structure
Expected ZIP structure:
```
dance_dataset.zip
├── class1
│   ├── img1.jpg
│   ├── ...
├── class2
│   ├── img1.jpg
│   ├── ...
...
```

## Requirements
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

Install with:
```bash
pip install -r requirements.txt
```

## License
MIT License