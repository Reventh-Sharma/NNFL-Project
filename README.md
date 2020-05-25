# NNFL-Project
Course project for Neural Networks and Fuzzy Logic, II Sem 2019-20

Title: Building a Convolutional Neural Network, using an Optimized Backpropagation Technique
  
Link to the problem statement:
https://bitsnnfl.github.io/cnn/optimization/backpropagation/object%20recognition/id-52-convolutional-neural-network-with-an-optimized-backpropagation-technique/

Paper Link: https://ieeexplore.ieee.org/document/8878719

## Requirements

* **Tensorflow 2.x** (Anything above Tensorflow 2.0)

## Instructions (Running the Python files directly)


*****************************************************************************************************************************************NOTE: The project file small_cnn_model_Project.ipynb trains a smaller CNN model for full batch (or fractional batch with minimum batch size to be 1/5th of training data size) calculations as was specified by iterations in paper52. This model, which was maximum that could be build on Google Colab for full batch calculations, used 11.7GBs of RAM when running for these high batch sizes provided in paper. For higher accuracy model (similiar to that in paper) trained on batch size = 20 please refer CNN_project.ipynb on which the presentation and .py files are based.
****************************************************************************************************************************************

Instructions for running the code directly:
1. Get the *COIL-100* dataset and place it on `./dataset/coil-100`, **OR** run this helper script, which will automatically do this for you:
```
python fetch_dataset.py
```

2. Train the model from scratch using:
```
python train.py
```

3. If you want to verify our results from the paper, you can directly load our pre-trained model's weights and test it using using:
```
python test.py N
```

Here, `N` represents the number of test images to sample from. By default, `N = 10`.

Otherwise, you can freshly train the model before testing, using:
```
python test.py N --train
```

## Group Members
* Vishnu Venkatesh
* R. Vijay Krishna
* Reventh Sharma
