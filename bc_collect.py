"""
Records training data for behavioral cloning. Drive the car yourself and
every frame is saved together with the current steering and throttle.

Output:
  data/frames/*.jpg   cropped 200x66 road images
  data/log.csv        frame_file, steering, throttle, timestamp

Click the preview window, then:
  R  start / stop recording
  Q  quit
"""

import csv
import os
import time

import cv2
import numpy as np
from beamngpy.sensors import Electrics

from config import CAM_WIDTH, CAM_HEIGHT, CROP_TOP, CROP_BOTTOM, DATA_DIR, FRAMES_DIR
from dave2 import crop_resize
from sim import open_beamng, wait_for_player_vehicle, front_camera, wait_for_frame

# Electrics reports the steering wheel angle in degrees
STEER_NORM = 500.0


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    bng = open_beamng()
    vehicle = wait_for_player_vehicle(bng)
    camera = front_camera(bng, vehicle, 'bc_cam')
    electrics = Electrics()
    vehicle.sensors.attach('bc_elec', electrics)
    wait_for_frame(camera)

    log_path = os.path.join(DATA_DIR, 'log.csv')
    is_new = not os.path.exists(log_path)
    # Continue numbering after any frames that are already there
    frame_idx = 0 if is_new else sum(1 for _ in open(log_path)) - 1
    log_file = open(log_path, 'a', newline='')
    log = csv.writer(log_file)
    if is_new:
        log.writerow(['frame_file', 'steering', 'throttle', 'timestamp'])

    recording = False
    saved = 0
    print("Click the preview window: R record, Q quit")

    while True:
        colour = camera.poll().get('colour')
        if colour is None:
            cv2.waitKey(1)
            continue
        frame = cv2.cvtColor(np.array(colour), cv2.COLOR_RGBA2BGR)

        vehicle.sensors.poll()
        e = electrics.data or {}
        steering = float(np.clip(float(e.get('steering', e.get('steering_input', 0.0))) / STEER_NORM, -1.0, 1.0))
        throttle = float(e.get('throttle', e.get('throttle_input', 0.0)))

        if recording:
            fname = f"{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(FRAMES_DIR, fname), crop_resize(frame), [cv2.IMWRITE_JPEG_QUALITY, 92])
            log.writerow([fname, f"{steering:.6f}", f"{throttle:.6f}", f"{time.time():.3f}"])
            log_file.flush()
            frame_idx += 1
            saved += 1

        preview = frame.copy()
        cv2.putText(preview, "REC" if recording else "STANDBY", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255) if recording else (180, 180, 180), 2)
        cv2.putText(preview, f"Steer: {steering:+.3f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, f"Throttle: {throttle:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(preview, f"Saved: {saved}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        # Crop lines, so you can see what the network will get
        for y in (int(CAM_HEIGHT * CROP_TOP), int(CAM_HEIGHT * CROP_BOTTOM)):
            cv2.line(preview, (0, y), (CAM_WIDTH, y), (0, 255, 255), 1)
        cv2.imshow('BC Collector', preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recording = not recording
            print(f"Recording {'started' if recording else 'stopped'} ({saved} frames so far)")

    cv2.destroyAllWindows()
    log_file.close()
    camera.remove()
    vehicle.sensors.detach('bc_elec')
    bng.disconnect()
    print(f"{saved} frames saved to {DATA_DIR}")


if __name__ == '__main__':
    main()
