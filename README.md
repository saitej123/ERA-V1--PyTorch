# Handwritten Digit Recognition for MNIST Dataset with PyTorch

This repository provides use PyTorch to train MNIST dataset
## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Files Details ( How To Read )](#files-details)
- [How To Use](#usage)
- [Results](#results)

## Introduction

This repository contains Python scripts for training and testing a Convolutional Neural Network (CNN) model on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) for recognizing handwritten digits. 

## Features

- Leveraging PyTorch for efficient computational execution
- Preprocessing the MNIST dataset utilizing data augmentation techniques
- Visualizing the processed images and their respective labels
- Training a Convolutional Neural Network model on the processed data
- Evaluating the trained model on the test portion of the data
- Plotting the accuracy and loss curves during training and testing


## Requirements

- Python 3.9+
- PyTorch 2.0
- torchvision
- Matplotlib
- CUDA (recommended for GPU)

## File Descriptions

- `utils.py`: This file contains all the helper functions required for the workflow, such as data preprocessing of the MNIST dataset, data visualization, model training and testing.

- `model.py`: This file defines the architecture of the Convolutional Neural Network model used for the task. It includes convolutional layers and fully connected layers.

- `S5.ipynb` : The main Jupyter notebook that manages the entire process. It imports functions from `utils.py` and the model from `model.py` to train and evaluate the model on the MNIST dataset. The notebook orchestrates the overall workflow.

## Usage

- Run the `S5.ipynb` notebook in a Jupyter notebook environment. If `CUDA` is available on your machine, the code will automatically use it to speed up computations.

- The notebook includes the following steps:

    1. **Data preprocessing**: Application of transformations on the MNIST dataset for data augmentation and normalization.
    2. **Model training**: Training the `CNN model` using the preprocessed training set.
    3. **Model testing**: Evaluation of the trained model on the test dataset and print of the loss and accuracy.
    4. **Performance visualization**: Plotting of accuracy and `loss graphs` for both the training and test datasets.

## Results

The final trained model achieves an accuracy of 99.80% on the MNIST test set.

```
Model Summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```

`Developed by Sai teja`