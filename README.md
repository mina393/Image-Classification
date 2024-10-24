# Parking Area Classification using SVM

This project uses a Support Vector Machine (SVM) classifier to determine whether a parking area is empty or not based on images. The model is trained on image data categorized as either `empty` or `not_empty` and evaluates performance through accuracy metrics.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training Workflow](#model-training-workflow)
- [Results](#results)
- [Future Improvements](#future-improvements)

## Overview

This project builds a classifier that analyzes parking lot images and predicts if they are empty or occupied. It uses an SVM model trained with images resized to 15x15 pixels to ensure efficient computation while maintaining reasonable accuracy. The trained model is saved for future predictions.

## Features

- **Support Vector Machine (SVM) classifier** for binary classification (empty vs. not_empty).
- **GridSearchCV** for hyperparameter tuning.
- **Image preprocessing**: Resizes all images to a fixed 15x15 pixel size.
- **Model saving**: Stores the trained model using `pickle` for future use.

## Requirements

- Python 3.x
- NumPy
- Scikit-learn
- Scikit-image (skimage)
- Pickle

You can install the required dependencies by running:
```bash
pip install numpy scikit-learn scikit-image
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/parking-area-classification.git
   ```
2. **Navigate to the project directory**:
   ```bash
   cd parking-area-classification
   ```
3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the image data**:
   - Organize the image dataset in two folders: `empty` for images where the parking area is empty, and `not_empty` for images with occupied parking spaces.
   - Set the path to your dataset folder in the `input_dir` variable inside the script.

2. **Run the classification script**:
   ```bash
   python main.py
   ```

3. **Model Output**:
   - The script will train the SVM model on the provided dataset.
   - It will print the classification accuracy as a percentage of correctly classified images.
   - The trained model will be saved as `model.pkl` for later use.

## Model Training Workflow

1. **Data Preprocessing**:
   - The image dataset is preprocessed by resizing each image to 15x15 pixels, which significantly reduces computation time.
   - The image data is flattened into 1D arrays before being passed into the classifier.

2. **Train/Test Split**:
   - The data is split into training and testing sets using an 80/20 ratio with stratified sampling to maintain the balance between classes.

3. **Grid Search**:
   - A grid search is performed to find the best hyperparameters for the SVM classifier. The grid searches over different values of `C`, `gamma`, and `kernel`.

4. **Model Evaluation**:
   - The trained model is evaluated on the test set, and the accuracy score is printed.

## Results

After training the model using the dataset, the classification accuracy is printed in the terminal. The final trained model is saved as `model.pkl` for use in real-time parking space detection systems or future testing.

## Future Improvements

- **Image Augmentation**: Apply image augmentation techniques (like rotation, flipping) to improve the model's robustness.
- **Improved Image Resolution**: Test higher image resolutions to increase classification accuracy.
- **Real-time Detection**: Integrate the model into a real-time parking lot monitoring system using live camera feeds.
