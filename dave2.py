"""
DAVE-2 steering network (Bojarski et al., "End to End Learning for
Self-Driving Cars", 2016) and the preprocessing used in training and driving.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from config import CROP_TOP, CROP_BOTTOM, IMG_W, IMG_H


class DaveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3),           nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3),           nn.ELU(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 3, IMG_H, IMG_W)).view(1, -1).shape[1]

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 100), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(100, 50),  nn.ELU(), nn.Dropout(0.1),
            nn.Linear(50, 10),   nn.ELU(),
            nn.Linear(10, 1),    nn.Tanh(),   # steering in [-1, 1]
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def crop_resize(frame):
    """Full camera frame (BGR) -> 200x66 road crop (BGR). Used when recording data."""
    h, w = frame.shape[:2]
    cropped = frame[int(h * CROP_TOP):int(h * CROP_BOTTOM), 0:w]
    return cv2.resize(cropped, (IMG_W, IMG_H))


def to_tensor(small_bgr, device):
    """200x66 BGR image -> normalized YUV tensor of shape (1, 3, 66, 200)."""
    yuv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YUV)
    return torch.from_numpy(yuv.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def preprocess(frame, device):
    return to_tensor(crop_resize(frame), device)


def load_model(path, device):
    model = DaveNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
