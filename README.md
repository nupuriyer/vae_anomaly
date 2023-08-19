# vae_anomaly
Variational autoencoder and decoder is a probabilistic set-up that allows users to create and evaluate synthetically generated information based on the input dataset. Simply put, a variational autoencoder tries to mimic the input dataset by fitting on it, while the decoder tries to identify the difference between the real data and that created by the autoencoder. 
The flow of this project will be as follows:
- Preprocessing: This includes converting the input video into an array of images.
- Fitting the model: We will then build the anomaly model based on the principle specified above and set up a threshold value fine-tuned to the video. This will depend on the input video. For this project the boat_river video is used.
