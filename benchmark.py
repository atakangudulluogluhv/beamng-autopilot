"""
Per-frame processing time of each perception stage on a single 800x600 frame.
Doesn't need BeamNG running.

Usage:
  python benchmark.py [--image first_frame.png] [--runs 300]
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

from config import ROOT, CAM_WIDTH, CAM_HEIGHT, MODEL_PATH
from dave2 import load_model, preprocess
from lane_detection import LaneDetector
from vehicle_detection import load_yolo, YOLO_ROI_BOTTOM


def time_ms(fn, runs, warmup=20, cuda=False):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return np.median(times), np.percentile(times, 95)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=os.path.join(ROOT, 'first_frame.png'),
                        help='800x600 camera frame, e.g. the one saved by camera_test.py')
    parser.add_argument('--runs', type=int, default=300)
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(f"{args.image} not found. Run camera_test.py to grab a frame.")
    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))

    lane = LaneDetector(CAM_WIDTH, CAM_HEIGHT)
    rows = [('Lane detection', 'CPU', *time_ms(lambda: lane.detect(frame), args.runs))]

    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for dev in devices:
        device = torch.device(dev)
        model = load_model(MODEL_PATH, device)
        with torch.no_grad():
            rows.append(('DAVE-2 (incl. preprocessing)', dev.upper(),
                         *time_ms(lambda: model(preprocess(frame, device)), args.runs, cuda=dev == 'cuda')))

    yolo = load_yolo()
    yolo_input = frame[:int(CAM_HEIGHT * YOLO_ROI_BOTTOM)]   # same crop as VehicleDetector
    for dev in devices:
        rows.append(('YOLO26n', dev.upper(),
                     *time_ms(lambda: yolo(yolo_input, verbose=False, device=dev), args.runs // 3,
                              cuda=dev == 'cuda')))

    print(f"\n{'stage':30s} {'device':6s} {'median ms':>10s} {'p95 ms':>8s}")
    for name, dev, med, p95 in rows:
        print(f"{name:30s} {dev:6s} {med:10.2f} {p95:8.2f}")


if __name__ == '__main__':
    main()
