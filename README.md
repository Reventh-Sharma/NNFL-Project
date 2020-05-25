# NNFL-Project
Course project for Neural Networks and Fuzzy Logic, II Sem 2019-20

Title: Building a Convolutional Neural Network, using an Optimized Backpropagation Technique
  
Link to the problem statement:
https://bitsnnfl.github.io/cnn/optimization/backpropagation/object%20recognition/id-52-convolutional-neural-network-with-an-optimized-backpropagation-technique/

Paper Link: https://ieeexplore.ieee.org/document/8878719

## Requirements

* **Tensorflow 2.x** (Anything above Tensorflow 2.0)

## Instructions (Running the Python files directly)

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
python test.py
```

Otherwise, you can freshly train the model before testing, using:
```
python test.py --train
```

## Group Members
* Vishnu Venkatesh
* R. Vijay Krishna
* Reventh Sharma