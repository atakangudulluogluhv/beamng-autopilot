"""
Trains the DAVE-2 network on data recorded with bc_collect.py.

The validation set is the last VAL_SPLIT of the recording in time order, not a
random sample. Frames are recorded at ~48 fps, so a random split would put
near-identical neighbours of training frames into validation.

Usage:
  python bc_train.py [--data DIR]

Output:
  models/bc_model.pth     best weights (lowest validation loss)
  models/bc_metrics.json  validation metrics of the best epoch
  docs/bc_train.png       loss curve
"""

import argparse
import csv
import json
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import ROOT, DATA_DIR, MODEL_PATH, IMG_W, IMG_H
from dave2 import DaveNet

EPOCHS     = 50
BATCH_SIZE = 64
LR         = 1e-4
VAL_SPLIT  = 0.15
STRAIGHT   = 0.10   # |steering| below this counts as driving straight
SEED       = 0

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DrivingDataset(Dataset):
    def __init__(self, frames_dir, samples, augment=False):
        self.frames_dir = frames_dir
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, steering = self.samples[idx]
        img = cv2.imread(os.path.join(self.frames_dir, fname))
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


def load_samples(data_dir):
    """(frame_file, steering) pairs in recording order."""
    log_path = os.path.join(data_dir, 'log.csv')
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No data at {log_path}. Record some with bc_collect.py first.")

    frames_dir = os.path.join(data_dir, 'frames')
    samples = []
    with open(log_path) as f:
        for row in csv.DictReader(f):
            fname = row['frame_file']
            if os.path.exists(os.path.join(frames_dir, fname)):
                samples.append((fname, float(np.clip(float(row['steering']), -1.0, 1.0))))
    samples.sort()
    return samples


def balance(samples):
    """Most driving is straight. Keep at most two straight frames per curve frame
    so the model doesn't learn to always go straight."""
    straight = [s for s in samples if abs(s[1]) < STRAIGHT]
    curved   = [s for s in samples if abs(s[1]) >= STRAIGHT]
    random.shuffle(straight)
    keep = min(len(straight), len(curved) * 2)
    return curved + straight[:keep], len(curved), keep


def evaluate(model, loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, steers in loader:
            preds.append(model(imgs.to(DEVICE)).cpu())
            labels.append(steers)
    return torch.cat(preds).numpy().ravel(), torch.cat(labels).numpy().ravel()


def metrics(pred, label):
    err = pred - label
    curve = np.abs(label) >= STRAIGHT
    # *_zero: the same metric for a model that always outputs 0 (drive straight)
    return {
        'mse':                float(np.mean(err ** 2)),
        'rmse':               float(np.sqrt(np.mean(err ** 2))),
        'rmse_zero':          float(np.sqrt(np.mean(label ** 2))),
        'mae':                float(np.mean(np.abs(err))),
        'mae_zero':           float(np.mean(np.abs(label))),
        'mae_curves':         float(np.mean(np.abs(err[curve]))),
        'mae_curves_zero':    float(np.mean(np.abs(label[curve]))),
        'mae_straight':       float(np.mean(np.abs(err[~curve]))),
        'curve_direction_acc': float(np.mean(np.sign(pred[curve]) == np.sign(label[curve]))),
        'r2':                 float(1 - np.sum(err ** 2) / np.sum((label - label.mean()) ** 2)),
    }


def train(data_dir):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"Device: {DEVICE}")

    samples = load_samples(data_dir)
    split = int(len(samples) * (1 - VAL_SPLIT))
    train_samples, n_curved, n_straight = balance(samples[:split])
    val_samples = samples[split:]
    print(f"{len(samples)} frames: train {split} -> {len(train_samples)} after balancing "
          f"({n_curved} curve, {n_straight} straight), val {len(val_samples)}")

    frames_dir = os.path.join(data_dir, 'frames')
    train_loader = DataLoader(DrivingDataset(frames_dir, train_samples, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(DrivingDataset(frames_dir, val_samples),
                              batch_size=BATCH_SIZE, shuffle=False)

    model     = DaveNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    best = None
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

        m = metrics(*evaluate(model, val_loader))
        scheduler.step(m['mse'])
        train_hist.append(t_loss)
        val_hist.append(m['mse'])
        print(f"epoch {epoch:3d}/{EPOCHS}  train {t_loss:.5f}  val {m['mse']:.5f}  "
              f"val MAE {m['mae']:.4f}  lr {optimizer.param_groups[0]['lr']:.1e}")

        if best is None or m['mse'] < best['mse']:
            best = dict(m, epoch=epoch)
            torch.save(model.state_dict(), MODEL_PATH)

    best.update(train_frames=len(train_samples), val_frames=len(val_samples))
    with open(os.path.join(os.path.dirname(MODEL_PATH), 'bc_metrics.json'), 'w') as f:
        json.dump(best, f, indent=2)

    plt.figure(figsize=(8, 3.5))
    plt.plot(range(1, EPOCHS + 1), train_hist, label='train (balanced, augmented)')
    plt.plot(range(1, EPOCHS + 1), val_hist, label='validation (last 15%, unbalanced)')
    plt.axvline(best['epoch'], color='grey', ls='--', lw=0.8)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(ROOT, 'docs', 'bc_train.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=120)

    print(f"Best epoch {best['epoch']}: val RMSE {best['rmse']:.4f} (always 0: {best['rmse_zero']:.4f}), "
          f"curve MAE {best['mae_curves']:.4f} (always 0: {best['mae_curves_zero']:.4f}), R2 {best['r2']:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=DATA_DIR, help='folder with log.csv and frames/')
    train(parser.parse_args().data)
