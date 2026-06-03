# Fracture Detection Using Vision Transformers

## Overview

This project presents an AI-powered system for automated musculoskeletal fracture classification and localization using the FracAtlas dataset.

The system combines:

- Vision Transformers (ViT) for fracture classification
- Faster R-CNN for fracture localization
- Streamlit deployment for interactive clinical use

The objective is to support clinicians by automatically identifying fracture presence and highlighting fracture regions in musculoskeletal X-ray images.

## Dataset

Dataset: FracAtlas Musculoskeletal Fracture Dataset

The dataset contains annotated musculoskeletal X-ray images including fracture and non-fracture cases, enabling both image classification and object detection tasks.

## Project Components

### Fracture Classification
A Vision Transformer model was trained to classify X-ray images as:

- Fracture
- Non-Fracture

### Fracture Localization
A Faster R-CNN object detection model was trained using bounding box annotations to localize fracture regions within X-ray images.

### Streamlit Application
A Streamlit-based interface was developed to allow clinicians to upload X-ray images and receive automated fracture localization predictions.

## Repository Structure

```text
docs/
images/
notebooks/
app.py
requirements.txt
README.md
