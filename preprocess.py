import numpy as np
import glob
import os
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import math

def generate_images(directory, batch_size, labels, shuffle, target_size, rescale=1./255, validation_split=0.3, fmt='png', train=True):
    # Batch Generator -> Generates Numpy arrays from corresponding images after scaling
    if not isinstance(target_size, tuple):
        raise TypeError("Expected Tuple")
    assert validation_split <= 0.5
    
    data_path = pathlib.Path(directory)
    images = list(data_path.glob("*." + fmt))
    num_images = len(images)
    
    # A Label for each image
    if train is True:
        print(labels.shape[0], num_images)
    assert num_images == labels.shape[0]

    idx = 0

    images = np.array(images)
    inds = np.arange(num_images)
    
    if shuffle:
        np.random.shuffle(inds)
        images, labels = images[inds], labels[inds]
        # random.shuffle(images)
    
    val_size = math.ceil(batch_size * (validation_split))
    train_size = batch_size - val_size
    
    while True:
        image_batch_train = np.zeros((train_size,) + target_size, dtype='float') # target_size is already a tuple
        label_batch_train = np.zeros((train_size, labels.shape[1]))
        image_batch_val = np.zeros((val_size,) + target_size, dtype='float')
        label_batch_val = np.zeros((val_size, labels.shape[1]))
        for i in range(train_size):
            if idx == num_images:
                idx = 0
                # random.shuffle(images)
                np.random.shuffle(inds)
                images, labels = images[inds], labels[inds]
            img = images[idx]
            image = np.asarray(Image.open(img)).reshape(target_size) * rescale
            image_batch_train[i] = image
            label_batch_train[i] = labels[idx]
            idx += 1
        for i in range(train_size, batch_size):
            if idx == num_images:
                # Reset index and shuffle if we reach the end of the dataset
                idx = 0
                # random.shuffle(images)
                np.random.shuffle(inds)
                images, labels = images[inds], labels[inds]
            img = images[idx]
            image = np.asarray(Image.open(img)).reshape(target_size) * rescale
            image_batch_val[i - train_size] = image
            label_batch_val[i - train_size] = labels[idx]
            idx += 1

        if train is True:
            print(f'Training Batch Size = {train_size}')
            print(f'Validation Batch Size = {batch_size - train_size}')
        
        # Yield a generator for both the training as well as the validation set of images
        yield image_batch_train, label_batch_train, image_batch_val, label_batch_val

def numpy_to_img(normalized_img, scale=255.0):
    # The original array is normalized, so scale it again
    # RGB Mode expects uint8
    image = Image.fromarray((normalized_img * scale).astype('uint8'))
    return image

def one_hot_encoder(label_dataset):
    """
    Takes a labelled dataset of the form: [ 0 0 0 ... 1 1 1 ... 100 100 100 ... ]
    and returns a one-hot encoding representation of the labelled data
    """
    unique_labels = np.unique(label_dataset)
    one_hot_labels = np.zeros((label_dataset.shape[0], unique_labels.shape[0]))
    label_dataset = np.repeat(label_dataset, unique_labels.shape[0]).reshape(-1, unique_labels.shape[0])
    one_hot_labels[np.where(label_dataset == unique_labels)] = 1
    return one_hot_labels