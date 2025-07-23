# Facial_Deepfake-video_detection
[![License: Custom](https://img.shields.io/badge/License-Custom-blue.svg)](LICENSE) [![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.7-red.svg)](https://pytorch.org/)

 

> *Deepfake detection* using CNN + LSTM models, served through Flask.  

*Repository:* https://github.com/Shivam-26102003/Deepfake_facial-video_detection   

---

## 🔖 Table of Contents

## 🔖 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Models](#-models)
- [Web Application](#-web-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🧐 About

This repository implements a *Facial Deepfake Detection* pipeline combining:

- *ResNeXt-50* CNN to extract spatial features from video frames.
- *LSTM* RNN to model temporal inconsistencies across frame sequences.
- A balanced multi-source dataset (FaceForensics++, DFDC, Celeb-DF).
- A Flask-based web interface for real-time video uploads and predictions.

---

## 📁 Dataset

This project uses a combination of publicly available deepfake datasets:

- 📦 *[FaceForensics++](https://github.com/ondyari/FaceForensics)* – A popular dataset for facial manipulation detection.
- 📦 *[Deepfake Detection Challenge Dataset (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge/data)* – Released by Facebook via Kaggle.
- 📦 *[Celeb-DF (v2)](https://github.com/yuezunli/Celeb-DF)* – High-quality deepfake dataset for research.

---

## ✨ Features

- *Spatial Feature Extraction*: ResNeXt-50 pretrained on ImageNet for robust embeddings.
- *Temporal Analysis*: LSTM captures frame-to-frame artifacts indicative of deepfakes.
- *Web UI*: Upload any video via browser and receive confidence scores instantly.
- *Modular Design*: Separate scripts for preprocessing, training, inference, and serving.

---

## 🛠 Technologies

- *Language*: Python 3.8+
- *Deep Learning*: PyTorch >=1.7
- *Web Framework*: Flask
- *Computer Vision*: OpenCV, face_recognition
- *Data Handling*: NumPy, Pandas
- *Visualization*: Matplotlib

---

## ⚙ Installation

1. *Clone the repo*
   bash
   git clone https://github.com/Shivam-26102003/Deepfake_facial-video_detection.git
   cd Deepfake_facial-video_detection

2. **Set up a virtual environment**
   bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
3. *Install dependencies*
   bash
   pip install -r requirements.txt
4. **Prepare models**
   Download pretrained ResNeXt-50 via PyTorch Hub.
   Place trained LSTM checkpoint in models/ directory.

## ▶ Usage
1. **Preprocess videos**
   bash
   python scripts/preprocess.py --input_dir data/raw --output_dir data/processed
2. *Train the model*
   bash
   python train.py --config configs/train.yaml
3. **Inference on a video**
   bash
   python predict.py --video path/to/video.mp4
4. *Start the web server*
   bash
   python server.py
   
## 📂 Project Structure
    bash
      Deepfake_facial-video_detection/
      ├── data/                   # raw and processed video datasets
      ├── models/                 # model checkpoints
      ├── scripts/                # preprocessing & training scripts
      │   ├── preprocess.py
      │   └── train.py
      ├── static/                 # CSS, JS, images for Flask app
      ├── templates/              # HTML templates for Flask app
      ├── server.py               # Flask server entry-point
      ├── requirements.txt
      ├── LICENSE
      └── README.md

## 🔍 Models
1. ResNeXt-50_32x4d: Extracts 2048-dimensional feature vectors per frame.
2. LSTM: Single-layer, hidden size 2048, dropout 0.4, followed by a classifier.
3. Classifier: FC → LeakyReLU → Softmax for binary real/fake prediction.

## 🌐 Web Application
1. Flask backend (server.py) accepts video uploads and returns predictions.
2. Templates: Simple HTML form for file upload and result display.
3. Static: CSS for styling and JS for frontend interactions.

📈 Results
Performance on Mixed Dataset (6000 videos):

   *Metric	    Value*
 1. Accuracy	  87.8%
 2. Precision    89.3%
 3. Recall	     86.5%
 4. F1-score     87.9%
