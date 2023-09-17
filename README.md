# CNN-Waste-Detector

The Waste Detection System is a computer vision-based application that identifies and classifies different types of waste materials, including cardboard, glass, metal, paper, and plastic. This system utilizes TensorFlow and OpenCV to create a waste detection model, preprocess data, train the model, and perform real-time waste classification from a live video feed. 


<p align="center">
  Model Testing with Recorded Data
</p>
<p align="center">
  <a href="https://www.youtube.com/watch?v=HT6F2x5samY">
    <img src="https://img.youtube.com/vi/HT6F2x5samY/0.jpg" alt="Video">
  </a>
</p>




## Project Overview

The project is divided into several key components, including data preprocessing, model creation, model training, and real-time waste classification using a webcam. Below, each component is explained in more detail.

## Data Preprocessing

In the data preprocessing phase, the project performs the following tasks:

- Creates a TensorFlow image dataset from a directory containing waste images.
- Converts labels in the dataset to one-hot encoded labels.
- Normalizes image data to ensure consistency.

## CNN Model

The Convolutional Neural Network (CNN) model is responsible for waste classification. Here's an overview of the model:

- Defines a CNN model architecture with convolutional layers, max-pooling layers, dropout layers, and dense layers.
- Compiles the model with the Adam optimizer and categorical cross-entropy loss.
- Trains the model on the preprocessed dataset.
- Evaluates the model's accuracy and other metrics.

## Model Testing

The model is tested on a test dataset to evaluate its performance. Key metrics, including precision, recall, and categorical accuracy, are calculated to assess the model's ability to classify waste materials accurately.

## Model Testing with Recorded Data

This part of the project demonstrates real-time waste classification using a webcam. The steps include:

- Capturing a live video feed from the camera.
- Resizing video frames to match the model's input size.
- Performing waste classification on each frame.
- Displaying the waste type and classification confidence on the video feed.
- Writing the processed video to an output file (optional).

<p align="center">
  Model Testing with Recorded Data
</p>
<p align="center">
  <a href="https://www.youtube.com/watch?v=HT6F2x5samY">
    <img src="https://img.youtube.com/vi/HT6F2x5samY/0.jpg" alt="Video">
  </a>
</p>

## Results



## Credits

The data used in this project was obtained from the [TrashNet Dataset](https://github.com/garythung/trashnet), created by Gary Thung and Mindy Yang. We would like to express our gratitude for making this valuable dataset available to the research community.

Please refer to the [original repository](https://github.com/garythung/trashnet) for more information about the dataset and its creators.


## Contact

mail: mikolajczachorowski260203@gmail.com
