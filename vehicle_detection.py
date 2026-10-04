"""YOLOv8 vehicle detection with distance from the depth camera."""

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from config import CAM_WIDTH, CAM_HEIGHT, YOLO_MODEL

YOLO_CONF       = 0.40
VEHICLE_CLASSES = {2, 3, 5, 7}   # COCO: car, motorcycle, bus, truck

# The bottom of the frame is our own hood, so YOLO only sees the part above this line.
YOLO_ROI_BOTTOM = 0.78
# Boxes wider than this are usually false positives, but keep it high enough
# that a car we are closing in on is not thrown away.
YOLO_MAX_BOX_W  = 0.80


def load_yolo(path=YOLO_MODEL):
    # PyTorch >= 2.6 defaults torch.load to weights_only=True, which can't load
    # the full model object stored in ultralytics .pt files. The file is the
    # official yolov8n checkpoint, so turn the check off just for this load.
    orig_load = torch.load

    def full_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return orig_load(*args, **kwargs)

    torch.load = full_load
    try:
        return YOLO(path)
    finally:
        torch.load = orig_load


class DepthEstimator:
    def __init__(self, near, far):
        self.near, self.far = near, far

    def to_metres(self, depth_img):
        # Depth comes back as an 8-bit image (sometimes RGBA); one channel is enough.
        d = np.array(depth_img)
        if d.ndim == 3:
            d = d[:, :, 0]
        d = d.astype(np.float32) / 255.0
        return self.near + d * (self.far - self.near)

    def distance_in_box(self, depth_m, x1, y1, x2, y2):
        # Stop above the hood so we don't measure our own car.
        y2 = min(y2, int(CAM_HEIGHT * YOLO_ROI_BOTTOM) - 5)
        if y2 <= y1 or x2 <= x1:
            return None
        roi   = depth_m[y1:y2, x1:x2]
        valid = roi[(roi > 1.0) & (roi < self.far * 0.95)]
        if len(valid) == 0:
            return None
        # 20th percentile: close to the nearest surface but not thrown off by a few noisy pixels.
        return float(np.percentile(valid, 20))


class VehicleDetector:
    def __init__(self, depth_near, depth_far, dist_emergency, dist_target):
        self.model = load_yolo()
        self.depth = DepthEstimator(depth_near, depth_far)
        self.dist_emergency = dist_emergency
        self.dist_target    = dist_target

    def detect(self, colour_frame, depth_frame, display):
        """Returns the distance (m) to the closest vehicle in our path, or None.
        Draws the boxes on `display`."""
        depth_m = self.depth.to_metres(depth_frame) if depth_frame is not None else None
        roi_y = int(CAM_HEIGHT * YOLO_ROI_BOTTOM)
        results = self.model(colour_frame[:roi_y, :], verbose=False, conf=YOLO_CONF)[0]

        closest_dist, closest_box = None, None
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (x2 - x1) / CAM_WIDTH > YOLO_MAX_BOX_W or y2 > roi_y - 10:
                continue

            dist = self.depth.distance_in_box(depth_m, x1, y1, x2, y2) if depth_m is not None else None

            # Only vehicles roughly in front of us count as the lead car.
            mid_x = (x1 + x2) / 2
            if CAM_WIDTH * 0.2 < mid_x < CAM_WIDTH * 0.8 and dist is not None:
                if closest_dist is None or dist < closest_dist:
                    closest_dist, closest_box = dist, (x1, y1, x2, y2, dist)

            label = f"{results.names[cls]} {float(box.conf[0]):.2f}"
            if dist:
                label += f"  {dist:.1f}m"
            cv2.rectangle(display, (x1, y1), (x2, y2), (150, 150, 150), 1)
            cv2.putText(display, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if closest_box:
            x1, y1, x2, y2, dist = closest_box
            colour = (0, 0, 255) if dist < self.dist_emergency else \
                     (0, 165, 255) if dist < self.dist_target else (0, 255, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), colour, 3)
            cv2.putText(display, f"{dist:.1f}m", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        return closest_dist
