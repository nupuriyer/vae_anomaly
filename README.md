# Variational Autoencoder (VAE) & Decoder

![image](https://github.com/nupuriyer/vae_anomaly/assets/69424215/0696b5f1-8968-4b19-87eb-bed586f4ae77)


Anomaly detection is a major application of machine learning that has helped individuals sift through a plethora of information in order to spot the odd element out. It is an accelerated technique of identifying common elements and highlighting the ones that don’t fit the ‘frame’ thus created in the process.

Variational autoencoder and decoder is a probabilistic set-up that allows users to create and evaluate synthetically generated information based on the input dataset. Simply put, a variational autoencoder tries to mimic the input dataset by fitting on it, while the decoder tries to identify the difference between the real data and that created by the autoencoder. 

Thus for such model setups, the elements that have more data will be encoded better and thus will lead to a higher error when put through the decoder. In other words, elements more common in the dataset will be encoded with higher accuracies and will be tough for the decoder to spot as fake.

Using the above principle, we can identify anomalies in a series structured data such as a video. By breaking a video into individual frames, and passing them through this autoencoder decoder setup, we can highlight those frames that stand out since they’re not like the others. These would be frames where an anomalous behaviour within the context of the video is caught.

The flow of this project will be as follows:



1. **Preprocessing:** This includes converting the input video into an array of images.
2. **Fitting the model:** We will then build the anomaly model based on the principle specified above and set up a threshold value fine-tuned to the video. This will depend on the input video. For this project the boat_river video is used.
