import os
import argparse
import glob
import pathlib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from preprocess import generate_images, one_hot_encoder
from models import generate_model, load_model
from train import train_model

from tensorflow.python.framework.ops import disable_eager_execution

# Disable eager execution
disable_eager_execution()

# Disable bullshit logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def test_model(model, data_path, num_samples):
    gen = generate_images(directory=str(data_path), batch_size=num_samples,
                            labels=CLASS_LABELS,
                            shuffle=True, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS),
                            validation_split=0, train=False)
    # We don't need any validation images, since this is test-time
    test_images, test_labels, _, _ = next(gen)
    # Make the prediction using the model
    predictions = model.predict(test_images)

    test_images = test_images.reshape(test_images.shape[0], -1)

    def convert_to_image(img_arr, target_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS), scale=255.0):
        # The original array is normalized, so scale it again
        # RGB Mode expects uint8
        image = Image.fromarray((img_arr.reshape(target_shape) * scale).astype('uint8'))
        return image
    return np.apply_along_axis(np.argmax, arr=predictions, axis=1), np.apply_along_axis(convert_to_image, arr=test_images, axis=1), test_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the Adagrad and RMSprop models')
    parser.add_argument('--train', action='store_true', help='Flag for training the models just before testing')
    parser.add_argument('test_samples', metavar='N', type=int, default=10,
                    help='The number of test samples')

    args = parser.parse_args()

    if args.test_samples <= 0:
        raise ValueError('Number of test samples must be a positive integer')
    
    num_test_samples = args.test_samples

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
    
    if args.train == True:
        batch_train, label_train, batch_val, label_val = next(gen)
        
        adadelta_model = generate_model('Adadelta')
        history = train_model(adadelta_model, 'Adadelta')

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig('adadelta_plot.png')

        loss, acc = adadelta_model.evaluate(batch_val,  label_val, verbose=2)
        print(f"Adadelta : Loss = {loss}, Validation Accuracy = {acc}")
        adadelta_model.save_weights('adadelta_model_weights.h5')

        rmsprop_model = generate_model('RMSprop')
        history = train_model(rmsprop_model, 'RMSprop')

        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.legend(loc='upper right')
        plt.savefig('adadelta_plot.png')

        loss, acc = rmsprop_model.evaluate(batch_val,  label_val, verbose=2)
        print(f"RMSprop : Loss = {loss}, Validation Accuracy = {acc}")
        rmsprop_model.save_weights('rmsprop_model_weights.h5')
    else:
        adadelta_model = load_model('Adadelta')
        rmsprop_model = load_model('RMSprop')

    # Evaluate the models
    predictions, test_images, test_labels = test_model(adadelta_model, data_path, num_samples=num_test_samples)

    # Example Images for testing the Model
    for prediction, test_label in zip(predictions, test_labels):
        print(f"Adedelta => Predicted Label: {prediction}")
        print(f"Adadelta => Actual Label: {np.argmax(test_label)}")

    predictions, test_images, test_labels = test_model(rmsprop_model, data_path, num_samples=num_test_samples)

    # Example Images for testing the Model
    for prediction, test_label in zip(predictions, test_labels):
        print(f"RMSprop => Predicted Label: {prediction}")
        print(f"RMSprop => Actual Label: {np.argmax(test_label)}")