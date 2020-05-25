import glob
import os
import pathlib
import math

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten, Dropout, Reshape

from tensorflow.python.framework.ops import disable_eager_execution

# Disable eager execution
disable_eager_execution()

# Disable bullshit logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from preprocess import generate_images, one_hot_encoder
from models import generate_model

def train_model(model, name):
    if name == 'Adadelta':
        history = model.fit(batch_train, label_train, epochs=20, batch_size=50,
                validation_data=(batch_val, label_val))
        model.save_weights('adadelta_model_weights.h5')
        return history
    elif name == 'RMSprop':
        history = model.fit(batch_train, label_train, epochs=20, batch_size=50,
            validation_data=(batch_val, label_val))
        model.save_weights('rmsprop_model_weights.h5')
        return history
    else:
        raise ValueError("Only Adadelta and RMSprop optimizer models are allowed")

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'dataset', 'coil-100')
    data_path = pathlib.Path(data_dir)

    # Path to all images
    images = pathlib.Path(data_dir).glob("*.png")
    image_data = list(images)

    CLASS_LABELS = list([int(''.join(item.name.split('_')[0].lstrip('obj'))) for item in image_data])
    CLASS_LABELS = np.array((CLASS_LABELS))
    CLASS_LABELS = one_hot_encoder(CLASS_LABELS)
    
    # Define parameters for batch generation
    BATCH_SIZE = len(image_data)
    IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128
    NUM_CHANNELS = 3 # RGB Image has 3 channels

    # Use the batch generator. This outputs (train_images, val_images)
    gen = generate_images(directory=str(data_path), batch_size=BATCH_SIZE,
                            labels=CLASS_LABELS,
                            shuffle=True, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                            validation_split=0.3)

    # Example generation. This generates a single batch of training and validation images
    batch_train, label_train, batch_val, label_val = next(gen)

    # Adadelta Model

    model = generate_model('Adadelta', NUM_LABELS=100)
    print(model.summary())

    history = train_model(model, 'Adadelta')

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('adadelta_plot.png')

    loss, acc = model.evaluate(batch_val,  label_val, verbose=2)
    print(f"Adadelta : Loss = {loss}, Validation Accuracy = {acc}")

    # RMSprop Model

    model = generate_model('RMSprop', NUM_LABELS=100)
    print(model.summary())

    history = train_model(model, 'RMSprop')

    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('rmsprop_plot.png')

    loss, acc = model.evaluate(batch_val,  label_val, verbose=2)
    print(f"RMSprop : Loss = {loss}, Validation Accuracy = {acc}")