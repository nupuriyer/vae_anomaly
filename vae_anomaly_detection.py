import os
import cv2
from glob import glob
import numpy as np
from PIL import Image
from sklearn.datasets import make_classification
# Data processing
import pandas as pd
import numpy as np
from collections import Counter
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Model and performance
import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# File Paths
img_folder = 'D:/Workspace/vae_anomaly/video_to_image'
img_folder_test = 'D:/Workspace/vae_anomaly/test_frames'


def convert_video_to_images(img_folder, filename='boat_river.avi'):
    """
    Converts the video file (boat_river.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.
    Arguments
    ---------
    filename : (string) file name (absolute or relative path) of video file.
    img_folder : (string) folder where the video frames will be
    stored as JPEG images.
    """
    # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')

    # Make sure that the abscense/prescence of path separator doesn't throw an
    # error.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    # print(img_folder)

    # Instantiate the video object.
    video = cv2.VideoCapture(filename)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")

    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break

    video.release()
    cv2.destroyAllWindows()

    if i:
        print(f'Video converted\n{i} images written to {img_folder}')


def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes and normalizes the extracted image frames from a folder.
    The images are returned both as a Numpy array of flattened images (i.e. the images with the 3-d shape (im_
    Arguments
    ---------
    img_dir : (string) the directory where the images are stored.
    im_width : (int) The desired width of the image.
    The default value works well.
    im_height : (int) The desired height of the image.
    The default value works well.
    Returns
    X : (numpy.array) An array of the flattened images.
    images : (list) A list of the resized images.
    """
    images = []
    fnames = sorted(glob(f'{img_dir}{os.path.sep}frame*.jpg'))

    for fname in fnames:
        im = Image.open(fname)

        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))

        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)

        # Close the PIL image once converted and stored.
        im.close()

    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))

    return X, images


# X.shape
# len(images)
# images[0]
# Synthetic dataset


def build_anomaly_model(
        img_folder,
        im_width=60,
        im_height=44,
        filename='assignment3_video.avi'):
    """
    Builds the anomaly detector model.
    It utilizes the helper functions convert_video_to_images and load_images
    Arguments
    ---------
    img_folder : (string) folder where the video frames will be
    im_width : (int) The desired width of the image.
    The default value works well.
    im_height : (int) The desired height of the image.
    The default value works well.
    filename : (string) file name (absolute or relative path) of video file.
    Returns
    autoencoder: trained autoencoder model for anomaly detection finetuned to the video in filename
    """
    convert_video_to_images(img_folder, filename)
    X, images = load_images(img_folder, im_width, im_height)

    # Input layer
    input = tf.keras.layers.Input(shape=(X.shape[1],))
    # Encoder layers
    encoder = tf.keras.Sequential([
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu')])(input)
    # Decoder layers
    decoder = tf.keras.Sequential([
        layers.Dense(8, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(7920, activation="sigmoid")])(encoder)
    # Create the autoencoder
    autoencoder = tf.keras.Model(inputs=input, outputs=decoder)

    autoencoder.compile(optimizer='adam', loss='mae')
    # Fit the autoencoder
    history = autoencoder.fit(X, X,
                              epochs=20,
                              batch_size=64,
                              # validation_data=(X_test, X_test),
                              shuffle=True)
    
    autoencoder.save('autoencoder.h5')

    return autoencoder


def anomalous_frames(
    img_dir_test,
    autoencoder=build_anomaly_model(
        img_folder,
        im_width=60,
        im_height=44,
        filename='assignment3_video.avi')):
    """
    Predicts if input frame is anomalous or non-anomalous
    Arguments
    ---------
    img_dir_test : (string) the directory where the test images are stored.
    autoencoder: trained autoencoder model for anomaly detection
    Returns: null
    Prints prediction results for all frames in the img_dir_test folder
    """
    transformed_input_frames, test_images = load_images(img_dir_test)

    # Predict anomalies/outliers in the training dataset
    oprediction = autoencoder.predict(transformed_input_frames)
    # Get the mean absolute error between actual and reconstruction/prediction
    oprediction_loss = tf.keras.losses.mae(
        oprediction, transformed_input_frames)
    # Check the prediction loss threshold for 10% of outliers
    loss_threshold = 0.04305718876421453
    print(
        f'The prediction loss threshold for 10% of outliers is {loss_threshold:.2f}')
    # # Visualize the threshold
    # sns.histplot(prediction_loss, bins=30, alpha=0.8)
    # plt.axvline(x=loss_threshold, color='orange')
    output = [0 if i < loss_threshold else 1 for i in oprediction_loss]
    fnames = glob(f'{img_dir_test}{os.path.sep}frame*.jpg')
    for i in range(transformed_input_frames.shape[0]):
        if output[i] == 1:
            print(fnames[i] + ": Anomalous")
        else:
            print(fnames[i] + ": Not Anomalous")


anomalous_frames(img_folder_test)
