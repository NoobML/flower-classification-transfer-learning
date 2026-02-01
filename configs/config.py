import torch
import os

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', '102 flower', 'flowers')


# Model hyperparameters
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_WORKERS = 2

# Training hyperparameters
CUSTOM_CNN_EPOCHS = 1  # Just 1 epoch for testing
PRETRAINED_EPOCHS = 1  # Just 1 epoch for testing
LEARNING_RATE = 0.001

# Scheduler
STEP_SIZE_CUSTOM = 5
STEP_SIZE_PRETRAINED = 3
GAMMA = 0.5

# Image preprocessing
IMG_SIZE = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Results
RESULTS_DIR = 'results/'
PLOTS_DIR = 'results/plots/'
METRICS_FILE = 'results/metrics.json'

# Models to train
PRETRAINED_MODELS = ['alexnet', 'vgg16', 'resnet18', 'mobilenet_v2', 'efficientnet_b0', 'xception']