# CNN-Waste-Detector

The Waste Detection System is a computer vision-based application that identifies and classifies different types of waste materials, including cardboard, glass, metal, paper, and plastic. This system utilizes PyTorch and OpenCV to create a waste detection model, preprocess data, train the model, and perform real-time waste classification from a live video feed.

<p align="center">
  Model Testing with Recorded Data
</p>
<p align="center">
  <a href="https://youtu.be/KIH_VGu_R4Y">
    <img src="https://img.youtube.com/vi/KIH_VGu_R4Y/0.jpg" alt="Video" width="512" height="384">
  </a>
</p>

## Installation

1. Clone repo
```bash
git clone https://github.com/Micz26/CNN-Waste-Detector.git
```
2. Create venv and install dependencies
```bash
poetry install
```
3. Install pre-commit
```bash
pre-commit install
```
4. Activate venv
```bash
poetry shell
```
5. [Optional] install dependecies for working with jupyter notebooks
```bash
poetry install --extras "jupyter"
```
6. [Optional] install dependecies for lint checking
```bash
poetry install --extras "lint"
```

## Scripts

After Installation, you can run scripts below. Specify arguments in <>.

1. Script for training my models (`BaseModel` or `ModelRegularized`)
```
python scripts/train_model.py <model> <epochs> <on_save> <save_path>
```
2. Script for fine-tuning vgg16
```
python scripts/fine_tune_vgg16.py <epochs> <on_save> <save_path>
```
3. Script for recording real-time predictions
```
python scripts/record.py <model> <model_path>
```
4. Script for oversamplnig empty images
```
python scripts/add_empty_images_to_data.py
```

## Project Overview

The project is divided into several key components, including eda, data preprocessing, model training. Additionaly i fined-tuned `vgg16` model on the data and presented it's perforemance in real-time waste classification using a webcam.

## Data

The data used in this project was obtained from the [TrashNet Dataset](https://github.com/garythung/trashnet). I enhanced the dataset by adding images labeled as 'nothing,' by taking few pictures of my desk and oversampling them. To ensure the model effectively learns to identify when there isn't any trash present.

## Models

I developed two models, `BaseModel` and `ModelRegularized`. Both models consist of six convolutional layers, each followed by ReLU activation, batch normalization, and max pooling, whcih are then followed by linear transformations. The primary difference between the two models is that ModelRegularized includes dropout layers to reduce overfitting. Additionally, I fine-tuned a pre-trained `VGG16` model on the dataset.

## Results

The `BaseModel` achieved an accuracy of 75%, while `ModelRegularized` reached 70%. `VGG16` significantly outperformed my models, achieving approximately 96% accuracy.

## Model Testing with Recorded Data

This part of the project demonstrates real-time waste classification using a webcam. You can watch it on youtube by clicking on the video:

<p align="center">
  Model Testing with Recorded Data
</p>
<p align="center">
  <a href="https://youtu.be/KIH_VGu_R4Y">
    <img src="https://img.youtube.com/vi/KIH_VGu_R4Y/0.jpg" alt="Video" width="512" height="384">
  </a>
</p>

## Credits

The data used in this project was obtained from the [TrashNet Dataset](https://github.com/garythung/trashnet), created by Gary Thung and Mindy Yang.

Please refer to the [original repository](https://github.com/garythung/trashnet) for more information about the dataset and its creators.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Contact

mail: mikolajczachorowski260203@gmail.com
