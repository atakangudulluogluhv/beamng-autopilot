"""
Trains the DAVE-2 network on data recorded with bc_collect.py.

Output:
  models/bc_model.pth   best weights (lowest validation loss)
  docs/bc_train.png     loss curve
"""

import csv
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import ROOT, DATA_DIR, FRAMES_DIR, MODEL_PATH, IMG_W, IMG_H
from dave2 import DaveNet

# Older recordings stored the raw wheel angle in degrees
STEER_NORM = 500.0

EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-4
VAL_SPLIT  = 0.15
AUGMENT    = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DrivingDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, steering = self.samples[idx]
        img = cv2.imread(os.path.join(FRAMES_DIR, fname))
        if img is None:
            img = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        if self.augment:
            # Mirror the image and the steering angle
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                steering = -steering
            # Brightness jitter on the Y channel
            img = img.astype(np.float32)
            img[:, :, 0] *= random.uniform(0.6, 1.4)
            img = np.clip(img, 0, 255).astype(np.uint8)
            # A little label noise helps against overfitting
            steering = float(np.clip(steering + random.gauss(0, 0.005), -1.0, 1.0))

        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return img, torch.tensor([steering], dtype=torch.float32)


def load_samples():
    log_path = os.path.join(DATA_DIR, 'log.csv')
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No data at {log_path}. Record some with bc_collect.py first.")

    samples = []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            fname = row['frame_file']
            raw = float(row['steering'])
            steering = float(np.clip(raw / STEER_NORM if abs(raw) > 1.0 else raw, -1.0, 1.0))
            if not fname.startswith('-') and os.path.exists(os.path.join(FRAMES_DIR, fname)):
                samples.append((fname, steering))
    print(f"Loaded {len(samples)} samples")

    # Most driving is straight. Undersample it so the model doesn't learn to always go straight.
    straight = [s for s in samples if abs(s[1]) < 0.10]
    curved   = [s for s in samples if abs(s[1]) >= 0.10]
    keep = min(len(straight), len(curved) * 2)
    random.shuffle(straight)
    balanced = curved + straight[:keep]
    random.shuffle(balanced)
    print(f"After balancing: {len(balanced)} ({len(curved)} curved, {keep} straight)")
    return balanced


def train():
    print(f"Device: {DEVICE}")
    samples = load_samples()
    split = int(len(samples) * (1 - VAL_SPLIT))
    train_loader = DataLoader(DrivingDataset(samples[:split], augment=AUGMENT),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(DrivingDataset(samples[split:]),
                              batch_size=BATCH_SIZE, shuffle=False)

    model     = DaveNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    best_val = float('inf')
    train_hist, val_hist = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0.0
        for imgs, steers in train_loader:
            imgs, steers = imgs.to(DEVICE), steers.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), steers)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for imgs, steers in val_loader:
                imgs, steers = imgs.to(DEVICE), steers.to(DEVICE)
                v_loss += criterion(model(imgs), steers).item()
        v_loss /= len(val_loader)

        scheduler.step(v_loss)
        train_hist.append(t_loss)
        val_hist.append(v_loss)
        print(f"epoch {epoch:3d}/{EPOCHS}  train {t_loss:.5f}  val {v_loss:.5f}  "
              f"lr {optimizer.param_groups[0]['lr']:.1e}")

        if v_loss < best_val:
            best_val = v_loss
            torch.save(model.state_dict(), MODEL_PATH)

    plt.figure(figsize=(10, 4))
    plt.plot(train_hist, label='train')
    plt.plot(val_hist, label='val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Behavioral Cloning Training')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(ROOT, 'docs', 'bc_train.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Best val loss {best_val:.5f}. Model: {MODEL_PATH}  Plot: {plot_path}")


if __name__ == '__main__':
    train()
