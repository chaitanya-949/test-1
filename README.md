## Image Classification 

This Jupyter Notebook implements an image classification pipeline to categorize material types in images. It utilizes Convolutional Neural Networks (CNNs) with TensorFlow and Keras.

# Requirements

Python 3.6+ (Ensure your Jupyter Notebook kernel uses the desired Python version)

TensorFlow (Install using pip install tensorflow)

Keras (Usually installed with TensorFlow)

OpenCV (cv2) (pip install opencv-python)

Pillow (PIL Fork) (pip install Pillow)

scikit-learn (pip install scikit-learn)

# Notebook Breakdown

The notebook consists of several code cells that perform the following steps:

Import Libraries: Import necessary libraries like TensorFlow, Keras, OpenCV, and others.
Load Data: Define a function to load image data from a directory structure with subfolders representing classes. This function iterates through the directory, identifies subfolders as classes, and creates a list of tuples containing image paths and corresponding labels.

Preprocess Images: Implement image preprocessing steps such as loading images using OpenCV, resizing them to a target size, and normalizing pixel values.

Split Data: Split the loaded data into training, validation, and test sets using stratified sampling (to maintain class distribution across sets).

Create CNN Model: Define a CNN architecture with convolutional layers, pooling layers, a flattening layer, a fully connected layer, and a softmax output layer. The number of classes in the output layer is determined by the number of unique classes in the data.

Compile Model: Compile the model with an optimizer (e.g., Adam), a loss function (e.g., sparse categorical crossentropy), and a metric (e.g., accuracy).

Train Model: Train the model on the training data with early stopping to prevent overfitting. The model is evaluated on the validation set during training to monitor performance.

Evaluate Model: Evaluate the model's performance on the unseen test set and report accuracy.

(Optional) Classification Report: Utilize classification_report from scikit-learn to generate a detailed report on the model's performance for each class.

# Running the Notebook

Open the Jupyter Notebook in your preferred environment (e.g., Jupyter Notebook web interface).

Ensure you have the required libraries installed (refer to the "Requirements" section).

Update the data_dir variable in a code cell to point to your dataset directory.

Run each code cell individually or use the "Run All" option to execute the entire notebook.

## Test Loss: 0.5476108193397522
## Test Accuracy: 0.8571428656578064

