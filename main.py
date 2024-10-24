import os  # Import os for directory and file handling
from skimage.io import imread  # Import imread for reading images
from skimage.transform import resize  # Import resize for resizing images
import numpy as np  # Import NumPy for numerical operations
from sklearn.metrics import accuracy_score  # Import accuracy_score to evaluate model performance
from sklearn.model_selection import train_test_split  # Import train_test_split for splitting data
from sklearn.model_selection import GridSearchCV  # Import GridSearchCV for hyperparameter tuning
from sklearn.svm import SVC  # Import Support Vector Classifier
import pickle  # Import pickle for saving the trained model

# Prepare data
input_dir = r'D:\Modern\projects\opencv_project\data\ParkingLotDetectorAndCounter-20241023T011859Z-001\ParkingLotDetectorAndCounter\clf-data\clf-data'
categories = ['empty', 'not_empty']  # Define the categories for classification

data = []  # Initialize a list to store image data
labels = []  # Initialize a list to store corresponding labels

# Loop through each category and load the images
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):  # Iterate through files in the category directory
        img_path = os.path.join(input_dir, category, file)  # Construct the full image path
        img = imread(img_path)  # Read the image

        # Resize the image to 15x15 pixels and store it in a variable
        img_resized = resize(img, (15, 15))

        # Flatten the resized image and append to data list
        data.append(img_resized.flatten())
        # Append the corresponding label (0 for empty, 1 for not_empty)
        labels.append(category_idx)

# Convert data and labels lists to NumPy arrays for processing
data = np.asarray(data)
labels = np.asarray(labels)

# Train / test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train classifier
classifier = SVC()  # Instantiate the Support Vector Classifier

# Define hyperparameters for Grid Search
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000], 'kernel': ['rbf']}]

# Initialize GridSearchCV with the classifier and parameters
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters)

# Fit the model using the training data
grid_search.fit(x_train, y_train)

# Test performance
best_estimator = grid_search.best_estimator_  # Get the best estimator from grid search

y_prediction = best_estimator.predict(x_test)  # Make predictions on the test set

# Calculate the accuracy of the model
score = accuracy_score(y_prediction, y_test)

# Print the classification accuracy
print('{}% of samples were correctly classified'.format(str(score * 100)))

# Saving the model
pickle.dump(best_estimator, open('./model.pkl', 'wb'))  # Save the trained model to a file
