import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_ROOT = os.path.join(BASE_DIR, "chest_xray")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR = os.path.join(DATA_ROOT, "val")
TEST_DIR = os.path.join(DATA_ROOT, "test")

# Image configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Small batch size for CPU training
CHANNELS = 3

# Training configuration
EPOCHS = 10  # Reduced for CPU demo, can be increased by user
LEARNING_RATE = 1e-4

# Model export paths
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
CUSTOM_MODEL_PATH = os.path.join(EXPORT_DIR, "custom_cnn_model.h5")
PRETRAINED_MODEL_PATH = os.path.join(EXPORT_DIR, "resnet50_model.h5")

# Create export directory if not exists
os.makedirs(EXPORT_DIR, exist_ok=True)
