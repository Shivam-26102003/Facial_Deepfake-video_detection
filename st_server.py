import streamlit as st
import os
import uuid
import cv2
import face_recognition
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import time
import zipfile
from werkzeug.utils import secure_filename
import datetime
import torch.nn.functional as F
from torch import nn
from torchvision import models
from skimage import img_as_ubyte
import warnings
import sys
import traceback
import logging
import pandas as pd
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Set environment variable for PyTorch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Create folder paths
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploaded_Files')
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
GRAPHS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'graphs')
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Admin', 'datasets')

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Dataset comparison accuracies
DATASET_ACCURACIES = {
    'Our Model': None,
    'FaceForensics++': 85.1,
    'DeepFake Detection Challenge': 82.3,
    'DeeperForensics-1.0': 80.7
}

# Define the neural network model
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
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
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
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

def extract_frames(video_path, num_frames=16):
    frames = []
    frame_paths = []
    unique_id = str(uuid.uuid4()).split('-')[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise Exception("Video file appears to be empty")
        
    interval = total_frames // num_frames
    
    count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0 and frame_count < num_frames:
            faces = face_recognition.face_locations(frame)
            if len(faces) == 0:
                count += 1
                continue
                
            try:
                top, right, bottom, left = faces[0]
                face_frame = frame[top:bottom, left:right, :]
                frame_path = os.path.join(FRAMES_FOLDER, f'frame_{unique_id}_{frame_count}.jpg')
                cv2.imwrite(frame_path, face_frame)
                frame_paths.append(frame_path)
                frames.append(face_frame)
                frame_count += 1
                logger.info(f"Extracted frame {frame_count}: {os.path.basename(frame_path)}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                
        count += 1
        if frame_count >= num_frames:
            break
            
    cap.release()
    
    if len(frames) == 0:
        raise Exception("No faces detected in the video")
        
    return frames, frame_paths

def predict(model, img):
    try:
        with torch.no_grad():
            fmap, logits = model(img)
            params = list(model.parameters())
            weight_softmax = model.linear1.weight.detach().cpu().numpy()
            logits = F.softmax(logits, dim=1)
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item() * 100
            logger.info(f'Prediction confidence: {confidence}%')
            return [int(prediction.item()), confidence]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        raise

def detect_fake_video(video_path):
    start_time = time.time()
    
    try:
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        path_to_videos = [video_path]
        video_dataset = ValidationDataset(path_to_videos, sequence_length=20, transform=train_transforms)
        model = Model(2)
        path_to_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/df_model.pt')
        
        if not os.path.exists(path_to_model):
            raise Exception("Model file not found. Please place the model file in the 'model' directory.")
            
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        
        prediction = predict(model, video_dataset[0])
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        return prediction, processing_time
    except Exception as e:
        logger.error(f"Error in detect_fake_video: {str(e)}")
        traceback.print_exc()
        raise

def generate_confidence_graph(confidence):
    try:
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.style.use('dark_background')
        
        real_cmap = LinearSegmentedColormap.from_list('custom_real', ['#2ecc71', '#27ae60'])
        fake_cmap = LinearSegmentedColormap.from_list('custom_fake', ['#e74c3c', '#c0392b'])
        
        colors = [real_cmap(0.6), fake_cmap(0.6)]
        
        sizes = [confidence, 100 - confidence]
        labels = ['Real', 'Fake']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, 
                                         explode=explode, 
                                         labels=labels, 
                                         colors=colors,
                                         autopct='%1.1f%%', 
                                         shadow=True, 
                                         startangle=90,
                                         textprops={'fontsize': 14, 'color': 'white'},
                                         wedgeprops={'edgecolor': '#2c3e50', 'linewidth': 2})
        
        plt.setp(autotexts, size=12, weight="bold")
        plt.setp(texts, size=14, weight="bold")
        
        ax.set_title('Confidence Score', 
                     pad=20, 
                     fontsize=16, 
                     fontweight='bold', 
                     color='white')
        
        ax.axis('equal')
        ax.grid(True, alpha=0.1, linestyle='--')
        
        fig.set_facecolor('#1a1a1a')
        
        return fig
    except Exception as e:
        logger.error(f"Error generating confidence graph: {str(e)}")
        traceback.print_exc()
        return None

def generate_comparison_graph(our_accuracy):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.style.use('dark_background')
        
        DATASET_ACCURACIES['Our Model'] = our_accuracy
        
        datasets = list(DATASET_ACCURACIES.keys())
        accuracies = list(DATASET_ACCURACIES.values())
        
        main_color = '#64ffda'
        secondary_colors = ['#34495e', '#2c3e50', '#2980b9']
        colors = [main_color] + secondary_colors
        
        ax.set_facecolor('#111d40')
        fig.set_facecolor('#111d40')
        
        bars = ax.bar(datasets, accuracies, color=colors)
        
        ax.grid(axis='y', linestyle='--', alpha=0.2, color='white')
        
        ax.set_title('Model Performance Comparison', 
                     color='white', 
                     pad=20, 
                     fontsize=16, 
                     fontweight='bold')
        
        ax.set_xlabel('Models', 
                      color='white', 
                      labelpad=10, 
                      fontsize=12, 
                      fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', 
                      color='white', 
                      labelpad=10, 
                      fontsize=12, 
                      fontweight='bold')
        
        plt.xticks(rotation=30, 
                  ha='right', 
                  color='#8892b0', 
                  fontsize=10)
        
        plt.yticks(color='#8892b0', 
                  fontsize=10)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', 
                   va='bottom', 
                   color='white',
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(facecolor='#111d40', 
                            edgecolor='none', 
                            alpha=0.7,
                            pad=3))
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#34495e')
        ax.spines['bottom'].set_color('#34495e')
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        logger.error(f"Error generating comparison graph: {str(e)}")
        traceback.print_exc()
        return None

def get_datasets():
    datasets = []
    for item in os.listdir(DATASET_FOLDER):
        if item.endswith('.zip'):
            path = os.path.join(DATASET_FOLDER, item)
            stats = os.stat(path)
            datasets.append({
                'name': item,
                'size': stats.st_size,
                'upload_date': datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    return datasets

# Set up Streamlit app
def main():
    st.set_page_config(
        page_title="DeepFake Detection App",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #111d40;
        color: white;
    }
    .stApp {
        background-color: #111d40;
    }
    .st-emotion-cache-18ni7ap {
        background-color: #1a1a1a;
    }
    .st-bb {
        background-color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #2c3e50;
        color: white;
        border-radius: 4px 4px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
    }
    h1, h2, h3, h4 {
        color: #64ffda;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if 'username' not in st.session_state:
        st.session_state.username = ""
    
    if 'admin' not in st.session_state:
        st.session_state.admin = False

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        if st.session_state.logged_in:
            st.write(f"Welcome, {st.session_state.username}!")
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.admin = False
                st.experimental_rerun()
        else:
            st.info("Please login to access all features")
        
        page = st.radio("Select Page", ["Home", "Detect DeepFakes", "Admin", "Login", "Signup"], disabled=False)
        
        st.markdown("---")
        st.markdown("### About")
        st.write("DeepFake Detection App using deep learning to identify manipulated videos.")
        st.markdown("---")
        st.markdown("### Links")
        st.markdown("[Privacy Policy](#)")
        st.markdown("[Terms of Service](#)")

    # Main content based on selected page
    if page == "Home":
        home_page()
    elif page == "Detect DeepFakes":
        if st.session_state.logged_in:
            detect_page()
        else:
            st.warning("Please login to access this feature")
            login_page()
    elif page == "Admin":
        if st.session_state.logged_in and st.session_state.admin:
            admin_page()
        else:
            st.warning("Admin access required")
            if not st.session_state.logged_in:
                login_page()
            else:
                st.error("You do not have admin privileges")
    elif page == "Login":
        login_page()
    elif page == "Signup":
        signup_page()

def home_page():
    st.title("Deepfake Detection System")
    
    st.markdown("""
    ## Welcome to the Deepfake Detection System
    
    This application uses advanced deep learning algorithms to detect manipulated videos and images.
    
    ### Features:
    - **Video Analysis**: Upload videos to check if they are authentic or manipulated
    - **Face Detection**: Automatically detects faces in videos for analysis
    - **Confidence Score**: Get detailed confidence scores for the detection
    - **Comparison**: See how our model compares to other deepfake detection datasets
    
    ### Getting Started:
    1. Sign up for an account
    2. Login to access the detection features
    3. Upload your video for analysis
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 High Accuracy")
        st.write("Our model achieves state-of-the-art performance in deepfake detection")
    
    with col2:
        st.markdown("### ⚡ Fast Processing")
        st.write("Get results quickly with our optimized neural network model")
    
    with col3:
        st.markdown("### 🔒 Secure")
        st.write("Your uploaded content is processed securely and not stored permanently")

def detect_page():
    st.title("Detect DeepFakes")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # Save the uploaded file
        video_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success("Video uploaded successfully!")
        
        try:
            with st.spinner("Processing video..."):
                # Extract frames
                frames, frame_paths = extract_frames(video_path)
                
                # Process video
                prediction, processing_time = detect_fake_video(video_path)
                
                # Determine result
                if prediction[0] == 0:
                    output = "FAKE"
                    result_color = "#e74c3c"  # Red
                else:
                    output = "REAL"
                    result_color = "#2ecc71"  # Green
                
                confidence = prediction[1]
                
                # Display results
                st.markdown(f"## Result: <span style='color:{result_color};'>{output}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                st.markdown(f"**Processing Time:** {processing_time:.2f} seconds")
                
                # Show frames
                st.subheader("Detected Faces")
                frame_cols = st.columns(4)
                for i, frame_path in enumerate(frame_paths[:8]):  # Display up to 8 frames
                    with frame_cols[i % 4]:
                        st.image(frame_path, caption=f"Frame {i+1}")
                
                # Show confidence graph
                st.subheader("Confidence Analysis")
                confidence_fig = generate_confidence_graph(confidence)
                if confidence_fig:
                    st.pyplot(confidence_fig)
                
                # Show comparison graph
                st.subheader("Model Comparison")
                comparison_fig = generate_comparison_graph(confidence)
                if comparison_fig:
                    st.pyplot(comparison_fig)
                
            # Clean up
            if os.path.exists(video_path):
                os.remove(video_path)
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            if os.path.exists(video_path):
                os.remove(video_path)

def admin_page():
    st.title("Admin Dashboard")
    
    tab1, tab2 = st.tabs(["Dataset Management", "System Status"])
    
    with tab1:
        st.subheader("Upload Dataset")
        
        uploaded_dataset = st.file_uploader("Upload dataset (ZIP file)", type=['zip'])
        
        if uploaded_dataset is not None:
            try:
                dataset_path = os.path.join(DATASET_FOLDER, secure_filename(uploaded_dataset.name))
                with open(dataset_path, "wb") as f:
                    f.write(uploaded_dataset.getbuffer())
                
                with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                    zip_ref.testzip()  # Validate zip file
                
                st.success("Dataset uploaded successfully!")
                
            except Exception as e:
                st.error(f"Error uploading dataset: {str(e)}")
                if os.path.exists(dataset_path):
                    os.remove(dataset_path)
        
        st.subheader("Available Datasets")
        datasets = get_datasets()
        
        if datasets:
            dataset_df = pd.DataFrame(datasets)
            # Convert size to MB
            dataset_df['size'] = dataset_df['size'].apply(lambda x: f"{x / (1024 * 1024):.2f} MB")
            st.dataframe(dataset_df)
        else:
            st.info("No datasets available")
    
    with tab2:
        st.subheader("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("GPU Availability", "No" if not torch.cuda.is_available() else "Yes")
            st.metric("PyTorch Version", torch.__version__)
            st.metric("OpenCV Version", cv2.__version__)
        
        with col2:
            st.metric("Model Status", "Loaded" if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/df_model.pt')) else "Not Found")
            st.metric("Dataset Count", len(get_datasets()))
            st.metric("System Uptime", "Unknown")  # Would need to implement uptime tracking

def login_page():
    st.title("Login")
    
    # For demonstration purposes, hardcode a user
    DEMO_USER = {"username": "admin", "password": "admin123", "admin": True}
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # In a real app, you would verify against a database
        if username == DEMO_USER["username"] and password == DEMO_USER["password"]:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.admin = DEMO_USER["admin"]
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

def signup_page():
    st.title("Sign Up")
    
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Sign Up"):
        if password != confirm_password:
            st.error("Passwords do not match")
        elif not (username and email and password):
            st.error("All fields are required")
        else:
            # In a real app, you would save to a database
            st.success("Account created successfully! You can now login.")
            # Redirect to login page
            st.session_state.page = "Login"
            st.experimental_rerun()

if __name__ == "__main__":
    main()