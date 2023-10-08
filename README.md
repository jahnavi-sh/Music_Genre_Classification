# Music_Genre_Classification

Certainly! Here's a detailed explanation of the code you provided:

## Understanding the Problem Statement

The objective of this project is to build a deep learning algorithm for automatically classifying different music genres from audio files. The dataset used for this task is the GTZAN genre classification dataset, which consists of 1000 audio files, each lasting 30 seconds. There are 10 music genres, each with 100 audio tracks in .wav format. The goal is to use low-level frequency and time domain features to perform this classification.

## Approach and Methodology

1. **Feature Extraction:** The first step in this process involves extracting features from the audio files. Here, the focus is on identifying linguistic parts and discarding noise. This is achieved through Mel Frequency Cepstral Coefficients (MFCCs), a widely used technique in audio and speech processing.

2. **Mel Frequency Cepstral Coefficients (MFCCs):**
   - **Dividing Audio Signals:** Audio signals are divided into smaller frames, typically 20-40 milliseconds long.
   - **Frequency Analysis:** Different frequencies present in these frames are identified.
   - **Noise Separation:** Linguistic frequencies are separated from noise.
   - **Discrete Cosine Transform (DCT):** DCT is applied to these frequencies to retain a specific sequence of frequencies with high information content while discarding noise.

3. **K-Nearest Neighbors (KNN) Classification:**
   - **Distance Calculation:** A function is defined to calculate the distance between feature vectors and find neighbors.
   - **Nearest Neighbors:** Nearest neighbors are found based on calculated distances.
   - **Voting Mechanism:** A voting mechanism is used among neighbors to classify the genre of the audio file.
  
4. **Model Evaluation:** The accuracy of the KNN classifier is evaluated using a test dataset, and the accuracy score is printed.

## Code Explanation

1. **Import Libraries:**
   - Libraries such as `numpy`, `scipy`, and `python_speech_features` are imported to handle mathematical operations, audio processing, and feature extraction, respectively.

2. **Feature Extraction and Dataset Preparation:**
   - MFCC features are extracted from the audio files and stored in a binary file (`my.dat`).
   - The dataset is divided into training and testing sets.

3. **KNN Model Training and Evaluation:**
   - The KNN algorithm is applied to the training set and evaluated using the test set.
   - Accuracy is calculated as the ratio of correct predictions to the total number of test samples and printed.

This code outlines the complete process of building a genre classification system for audio files using MFCC features and KNN classification.
