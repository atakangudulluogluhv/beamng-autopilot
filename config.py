"""Settings shared by all scripts."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# BeamNG.tech install folder. Set the BNG_HOME environment variable to override.
BNG_HOME   = os.environ.get('BNG_HOME', r'C:\BeamNG.tech.v0.38.3.0')
BNG_BINARY = 'Bin64/BeamNG.tech.x64.exe'   # skip the launcher, start the game directly
HOST, PORT = 'localhost', 64256

# Front camera, mounted roughly at the driver's eye height
CAM_WIDTH, CAM_HEIGHT = 800, 600
CAM_POS = (0, 0.0, 1.8)
CAM_DIR = (0, -1, 0)
CAM_FOV = 70

# Behavioral cloning input: crop the road area, resize to the DAVE-2 input size
CROP_TOP     = 0.38   # drop sky / buildings
CROP_BOTTOM  = 0.95   # drop the hood
IMG_W, IMG_H = 200, 66

DATA_DIR   = os.path.join(ROOT, 'data')
FRAMES_DIR = os.path.join(DATA_DIR, 'frames')
MODEL_PATH = os.path.join(ROOT, 'models', 'bc_model.pth')
YOLO_MODEL = 'yolov8n.pt'   # downloaded by ultralytics on first run
