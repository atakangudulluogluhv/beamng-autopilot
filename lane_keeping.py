"""
Lane keeping assist using only the geometric lane detector and a PID controller.
Throttle is set by hand. Every control step is logged to data/lane_keeping_log.csv
for tuning the PID gains.

Click the camera window, then:
  A    toggle lane keeping
  W/S  throttle up / down
  D    debug view
  Q    quit
"""

import csv
import os
import time

import cv2
import numpy as np

from config import CAM_WIDTH, CAM_HEIGHT, DATA_DIR
from lane_detection import LaneDetector
from sim import PID, open_beamng, wait_for_player_vehicle, front_camera, wait_for_frame

MAX_STEER      = 0.60
MAX_STEER_RATE = 0.03
THROTTLE_BASE  = 0.15
CTRL_INTERVAL  = 0.10

# KD is effectively off: the derivative of a noisy vision signal just adds jitter.
KP, KI, KD = 0.0044, 0.000018, 0.0000001
ERROR_EMA  = 0.15   # heavy smoothing on the pixel error

LOG_COLUMNS = [
    'session_time', 'timestamp', 'active',
    'raw_error', 'smooth_error',
    'pid_p', 'pid_i', 'pid_d', 'pid_output',
    'target_steer', 'current_steer',
    'throttle', 'active_throttle',
    'confidence', 'lane_width_px',
    'kp', 'ki', 'kd',
]


def open_log():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, 'lane_keeping_log.csv')
    try:
        f = open(path, 'a', newline='')
    except PermissionError:
        # Usually means the file is open in Excel
        path = os.path.join(DATA_DIR, f"lane_keeping_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        f = open(path, 'a', newline='')
    writer = csv.writer(f)
    if os.path.getsize(path) == 0:
        writer.writerow(LOG_COLUMNS)
    return f, writer, path


def draw_confidence(img, conf):
    w = img.shape[1]
    col = (0, 255, 0) if conf > 0.6 else (0, 165, 255) if conf > 0.3 else (0, 0, 255)
    cv2.rectangle(img, (w - 130, 8), (w - 10, 22), (50, 50, 50), -1)
    if conf > 0:
        cv2.rectangle(img, (w - 130, 8), (w - 130 + int(120 * conf), 22), col, -1)
    cv2.putText(img, f"CONF {conf:.0%}", (w - 130, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


def main():
    bng = open_beamng()
    vehicle = wait_for_player_vehicle(bng)
    camera = front_camera(bng, vehicle, 'lka_cam')
    wait_for_frame(camera)

    detector = LaneDetector(CAM_WIDTH, CAM_HEIGHT)
    pid = PID(KP, KI, KD)
    active = False
    debug_view = False
    throttle = THROTTLE_BASE
    current_steer = 0.0
    smooth_error = 0.0
    prev_time = time.time()
    last_ctrl_t = 0.0

    log_file, log, log_path = open_log()
    session = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Logging to {log_path}")
    print("Click the camera window: A lane keeping, W/S throttle, D debug, Q quit")

    while True:
        colour = camera.poll().get('colour')
        if colour is None:
            cv2.waitKey(1)
            continue
        frame = cv2.cvtColor(np.array(colour), cv2.COLOR_RGBA2BGR)

        now = time.time()
        dt = max(now - prev_time, 1e-3)
        prev_time = now

        error, conf, annotated, colour_mask, roi_edges = detector.detect(frame)

        if active and (now - last_ctrl_t) >= CTRL_INTERVAL:
            if error is not None:
                if abs(error) < 8:
                    error = 0
                smooth_error = ERROR_EMA * error + (1 - ERROR_EMA) * smooth_error
                raw_steer = pid.step(smooth_error, dt)

                target_steer = float(np.clip(raw_steer, -MAX_STEER, MAX_STEER))
                current_steer += float(np.clip(target_steer - current_steer, -MAX_STEER_RATE, MAX_STEER_RATE))

                # Ease off the throttle in curves
                active_throttle = throttle * max(0.0, 1.0 - abs(smooth_error) / 80.0)
                vehicle.control(steering=current_steer, throttle=active_throttle, brake=0.0)

                log.writerow([
                    session, f"{now:.3f}", int(active),
                    f"{error:.1f}", f"{smooth_error:.2f}",
                    f"{pid.last_p:.5f}", f"{pid.last_i:.5f}", f"{pid.last_d:.5f}", f"{raw_steer:.5f}",
                    f"{target_steer:.4f}", f"{current_steer:.4f}",
                    f"{throttle:.3f}", f"{active_throttle:.3f}",
                    f"{conf:.2f}", detector.lane_width_px,
                    KP, KI, KD,
                ])
            else:
                # Lost the lane: let the wheel drift back to centre
                current_steer *= 0.9
                vehicle.control(steering=current_steer, throttle=throttle, brake=0.0)
            last_ctrl_t = now

        status_col = (0, 255, 0) if active else (0, 0, 255)
        cv2.putText(annotated, f"LKA: {'ON' if active else 'OFF'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_col, 2)
        if error is not None:
            cv2.putText(annotated, f"Err: {error:+.0f}px", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(annotated, f"Steer: {current_steer:+.2f}", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        cv2.putText(annotated, f"Throttle: {throttle:.2f}", (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.putText(annotated, "LANE OK" if error is not None else "SEARCHING", (10, 155),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if error is not None else (0, 100, 255), 1)
        draw_confidence(annotated, conf)
        cv2.imshow('Lane Keeping', annotated)

        if debug_view:
            dbg = np.hstack([cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR),
                             cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)])
            cv2.imshow('Debug: colour mask | ROI edges', cv2.resize(dbg, (CAM_WIDTH, CAM_HEIGHT // 2)))
        else:
            try:
                cv2.destroyWindow('Debug: colour mask | ROI edges')
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            active = not active
            pid.reset()
            current_steer = smooth_error = 0.0
            if not active:
                vehicle.control(steering=0.0, throttle=0.0, brake=0.0)
            print(f"Lane keeping {'on' if active else 'off'}")
        elif key == ord('w'):
            throttle = min(throttle + 0.05, 1.0)
        elif key == ord('s'):
            throttle = max(throttle - 0.05, 0.0)
        elif key == ord('d'):
            debug_view = not debug_view

    cv2.destroyAllWindows()
    log_file.close()
    camera.remove()
    vehicle.control(steering=0.0, throttle=0.0, brake=0.0)
    bng.disconnect()


if __name__ == '__main__':
    main()
