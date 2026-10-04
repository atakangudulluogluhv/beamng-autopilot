"""
Drives using only the behavioral cloning network. Useful for checking what the
model learned on its own, without the lane detector.

Click the camera window, then:
  A    toggle
  W/S  throttle up / down
  Q    quit
"""

import os

import cv2
import numpy as np
import torch

from config import CAM_WIDTH, CAM_HEIGHT, CROP_TOP, CROP_BOTTOM, MODEL_PATH
from dave2 import load_model, preprocess
from sim import open_beamng, wait_for_player_vehicle, front_camera, wait_for_frame

THROTTLE_BASE  = 0.15
MAX_STEER      = 0.30
MAX_STEER_RATE = 0.025

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}. Run bc_train.py first.")
        return
    model = load_model(MODEL_PATH, DEVICE)

    bng = open_beamng()
    vehicle = wait_for_player_vehicle(bng)
    camera = front_camera(bng, vehicle, 'bc_cam')
    wait_for_frame(camera)
    print("Click the camera window: A toggle, W/S throttle, Q quit")

    active = False
    throttle = THROTTLE_BASE
    current_steer = 0.0

    while True:
        colour = camera.poll().get('colour')
        if colour is None:
            cv2.waitKey(1)
            continue
        frame = cv2.cvtColor(np.array(colour), cv2.COLOR_RGBA2BGR)

        with torch.no_grad():
            raw_steer = float(model(preprocess(frame, DEVICE)).item())

        if active:
            target = float(np.clip(raw_steer, -MAX_STEER, MAX_STEER))
            current_steer += float(np.clip(target - current_steer, -MAX_STEER_RATE, MAX_STEER_RATE))
            vehicle.control(steering=current_steer, throttle=throttle, brake=0.0)

        hud = frame.copy()
        cv2.putText(hud, f"BC: {'ON' if active else 'OFF'}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if active else (0, 0, 255), 2)
        cv2.putText(hud, f"Model steer: {raw_steer:+.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(hud, f"Applied:     {current_steer:+.3f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(hud, f"Throttle:    {throttle:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        for y in (int(CAM_HEIGHT * CROP_TOP), int(CAM_HEIGHT * CROP_BOTTOM)):
            cv2.line(hud, (0, y), (CAM_WIDTH, y), (0, 255, 255), 1)

        bar_cx = CAM_WIDTH // 2
        col = (0, 255, 0) if abs(raw_steer) < 0.3 else (0, 165, 255) if abs(raw_steer) < 0.6 else (0, 0, 255)
        cv2.rectangle(hud, (bar_cx - 200, CAM_HEIGHT - 25), (bar_cx + 200, CAM_HEIGHT - 10), (50, 50, 50), -1)
        cv2.rectangle(hud, (bar_cx, CAM_HEIGHT - 25), (int(bar_cx + raw_steer * 200), CAM_HEIGHT - 10), col, -1)
        cv2.line(hud, (bar_cx, CAM_HEIGHT - 30), (bar_cx, CAM_HEIGHT - 5), (255, 255, 255), 1)
        cv2.imshow('BC Driver', hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            active = not active
            current_steer = 0.0
            if not active:
                vehicle.control(steering=0.0, throttle=0.0, brake=0.0)
        elif key == ord('w'):
            throttle = min(throttle + 0.05, 1.0)
        elif key == ord('s'):
            throttle = max(throttle - 0.05, 0.0)

    cv2.destroyAllWindows()
    camera.remove()
    vehicle.control(steering=0.0, throttle=0.0, brake=0.0)
    bng.disconnect()


if __name__ == '__main__':
    main()
