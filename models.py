import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Dense, Activation, Conv2D, AveragePooling2D, Flatten, Dropout, Reshape

def simple_model(optimizer, NUM_LABELS):
    model = models.Sequential()
    model.add(layers.Conv2D(8, (1, 1), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(12, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (1, 1), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_LABELS, activation='softmax'))
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return model

def complex_model(optimizer, NUM_LABELS):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(NUM_LABELS, activation='softmax'))
    model.compile(optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
    return model

def generate_model(optimizer, NUM_LABELS=100, do_minibatch=True):
    if do_minibatch == True:
        if optimizer == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=0.02, rho=0.96, epsilon=1e-03, name='Adadelta')
            return complex_model(optimizer, NUM_LABELS)
        elif optimizer == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, name='RMSprop')
            return complex_model(optimizer, NUM_LABELS)
        else:
            raise ValueError("Only Adadelta and RMSprop optimizers are allowed")
    else:
        if optimizer == 'Adadelta':
            optimizer = tf.keras.optimizers.Adadelta(
                learning_rate=0.02, rho=0.96, epsilon=1e-03, name='Adadelta')
            return simple_model(optimizer, NUM_LABELS)
        elif optimizer == 'RMSprop':
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-07, name='RMSprop')
            return simple_model(optimizer, NUM_LABELS)
        else:
            raise ValueError("Only Adadelta and RMSprop optimizers are allowed")


def load_model(type):
    if type == "Adadelta":
        model = generate_model("Adadelta")
        model.load_weights('adadelta_model_weights.h5')
        return model
    elif type == "RMSprop":
        model = generate_model("RMSprop")
        model.load_weights('rmsprop_model_weights.h5')
        return model
    else:
        raise ValueError("Only Adagrad and RMSprop models supported")