# Facial Deepfake Detection  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]() [![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D1.7-red.svg)]()  

> **Real-time deepfake detection** using CNN + LSTM models, served through Flask.  

**Repository:** https://github.com/Shivam-26102003/Deepfake_facial-video_detection  
**Demo (optional):** _https://your-demo-link.com_  

---

## 🔖 Table of Contents

- [About](#about)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Web Application](#web-application)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 🧐 About

This repository implements a **Facial Deepfake Detection** pipeline combining:

- **ResNeXt-50** CNN to extract spatial features from video frames.
- **LSTM** RNN to model temporal inconsistencies across frame sequences.
- A balanced multi-source dataset (FaceForensics++, DFDC, Celeb-DF).
- A Flask-based web interface for real-time video uploads and predictions.

---

## ✨ Features

- **Spatial Feature Extraction**: ResNeXt-50 pretrained on ImageNet for robust embeddings.
- **Temporal Analysis**: LSTM captures frame-to-frame artifacts indicative of deepfakes.
- **Web UI**: Upload any video via browser and receive confidence scores instantly.
- **Modular Design**: Separate scripts for preprocessing, training, inference, and serving.

---

## 🛠️ Technologies

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch >=1.7
- **Web Framework**: Flask
- **Computer Vision**: OpenCV, face_recognition
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib

---

## ⚙️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/Shivam-26102003/Deepfake_facial-video_detection.git
   cd Deepfake_facial-video_detection
"# Deepfake_facial-video_detection"
 
Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

