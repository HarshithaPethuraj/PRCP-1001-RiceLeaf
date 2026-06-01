---
title: Rice Leaf Disease Detector
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: streamlit
sdk_version: 1.39.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# 🌾 Rice Leaf Disease Detector

A CNN-based rice leaf disease classifier with Grad-CAM explainability,
deployed on Hugging Face Spaces.

## What it does

Upload a photo of a rice leaf and the app:
1. Predicts the disease — one of **Bacterial leaf blight**, **Brown spot**, or **Leaf smut**
2. Shows a **Grad-CAM heatmap** highlighting which leaf regions drove the prediction
3. Suggests **basic treatment guidance** for the predicted disease

## Model

- **Architecture:** MobileNetV2 (ImageNet pretrained) + 3-class softmax head
- **Trainable parameters:** 3,843 (frozen backbone — 2.26M total)
- **Input size:** 224 × 224 RGB
- **Disk size:** ~9 MB (TFLite-convertible to ~2–3 MB)
- **Performance** (PRCP-1001 dataset, 5-fold stratified CV):
  - Mean accuracy: 0.89
  - Final validation accuracy: 0.933
  - Macro-F1: 0.93
  - Macro ROC-AUC: 0.997

## Dataset

PRCP-1001 Rice Leaf Disease — 119 images across 3 classes (~40 each).
Trained with light augmentation (horizontal flip + small rotation) which
outperformed both no-augmentation and heavy-augmentation in controlled
experiments.

## Tech stack

- TensorFlow / Keras for model
- Gradio for the web UI
- Matplotlib jet colormap for the Grad-CAM overlay

## Author

**Harshitha Pethuraj** · Team ID PTID-AIE-MAY-26-11194
Project 4 of 4 in the AIE Capstone — Agriculture Domain.

## Disclaimer

This is an academic/demo tool. Always confirm any plant-disease diagnosis
with a local agricultural extension officer before applying pesticides or
fungicides.
