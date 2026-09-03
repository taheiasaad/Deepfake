# DeepFakeGuard

A deep learning project for detecting deepfake videos by classifying face images as REAL or FAKE.

## Overview

This project builds a binary image classifier that determines whether a face in a video frame is authentic or has been manipulated using deepfake techniques. It uses the Celeb-DF-v2 dataset, which contains real and AI-generated fake celebrity videos.

## How It Works

1. **Data Collection**: Videos are downloaded from the Celeb-DF-v2 dataset and split into train/validation/test sets at the video level (to avoid data leakage between frames of the same video).
2. **Frame Extraction**: A fixed number of frames are sampled evenly from each video using OpenCV.
3. **Face Detection**: MTCNN (a deep learning face detector) locates and crops the face in each frame, with a margin added around the box.
4. **Data Balancing**: Since fake videos outnumber real ones, real face images are augmented (flips, blur, noise, compression) to increase their count, and the fake class is undersampled to reach a balanced training set.
5. **Model**: A pretrained EfficientNetB0 (transfer learning) is used as the backbone, with a custom classification head added on top.
6. **Training**: Two-stage training — first the base model is frozen and only the new head is trained, then the last layers of EfficientNet are unfrozen and fine-tuned at a lower learning rate.
7. **Threshold Optimization**: Instead of using a fixed 0.5 threshold, the optimal decision threshold is selected based on the best F1-score on the validation set.
8. **Evaluation**: The model is evaluated on the test set using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and ROC curve.
9. **Inference**: Functions are provided to run predictions on a single image or an entire video (by averaging predictions across sampled frames).

## Tech Stack

- TensorFlow / Keras
- OpenCV
- MTCNN
- Albumentations
- scikit-learn
- pandas, numpy, matplotlib, seaborn

## Task Type

This is an image classification task (REAL vs FAKE), not object detection or segmentation. Face detection is used only as a preprocessing step to crop the region of interest before classification.

## Results

| Metric | Value |
|---|---|
| Accuracy | 0.89 |
| Precision | 0.90 |
| Recall | 0.99 |
| F1-Score | 0.94 |
| ROC-AUC | 0.82 |

## Dataset

[Celeb-DF-v2](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2) on Kaggle.
