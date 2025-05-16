from flask import Flask, render_template, redirect, request, url_for, send_file, send_from_directory
from flask import jsonify, json
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time
import uuid
import sys

# 'nn' Help us in creating & training of neural network
from torch import nn
import torch.nn.functional as F

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

# Get the absolute path for the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploaded_Files')
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)

# Ensure frames folder has proper permissions
os.chmod(FRAMES_FOLDER, 0o755)

video_path = ""
detectOutput = []

app = Flask("__main__", template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Creating Model Architecture
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained= True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size*seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm,_ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

def extract_frames(video_path, num_frames=8):
    frames = []
    frame_paths = []
    session_id = str(uuid.uuid4())[:8]
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames
    
    count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0 and frame_count < num_frames:
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                face_frame = frame[top:bottom, left:right, :]
                frame_path = os.path.join(FRAMES_FOLDER, f'frame_{session_id}_{frame_count}.jpg')
                cv2.imwrite(frame_path, face_frame)
                frame_paths.append(os.path.basename(frame_path))
                frames.append(face_frame)
                frame_count += 1
            except:
                pass
                
        count += 1
        if frame_count >= num_frames:
            break
            
    cap.release()
    return frames, frame_paths

def predict(model, img, path='./'):
    fmap, logits = model(img.to())
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = F.softmax(logits, dim=1)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item()*100
    print('confidence of prediction: ', confidence)
    return [int(prediction.item()), confidence]

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length = 60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0,a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top,right,bottom,left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def detectFakeVideo(videoPath):
    start_time = time.time()
    
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size,im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std)
    ])
    
    path_to_videos = [videoPath]
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
    model = Model(2)
    path_to_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/df_model.pt')
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    
    prediction = predict(model, video_dataset[0], './')
    
    processing_time = time.time() - start_time
    
    return prediction, processing_time

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return render_template('detect.html')
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('detect.html', error="No video file uploaded")
            
        video = request.files['video']
        if video.filename == '':
            return render_template('detect.html', error="No video file selected")
            
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return render_template('detect.html', error="Invalid file format. Please upload MP4, AVI, or MOV files.")
            
        video_filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)
        
        try:
            # Extract frames before prediction
            frames, frame_paths = extract_frames(video_path)
            
            # Get prediction and processing time
            prediction, processing_time = detectFakeVideo(video_path)
            
            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
            confidence = prediction[1]
            
            data = {
                'output': output, 
                'confidence': confidence,
                'frames': frame_paths,
                'processing_time': round(processing_time, 2)
            }
            data = json.dumps(data)
            
            os.remove(video_path)
            return render_template('detect.html', data=data)
            
        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            return render_template('detect.html', error=f"Error processing video: {str(e)}")

if __name__ == '__main__':
    app.run(port=3000, debug=False)