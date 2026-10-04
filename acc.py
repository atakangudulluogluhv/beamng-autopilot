"""
Adaptive cruise control. YOLOv8 finds the lead vehicle, the depth camera gives
the distance, and a PD controller on (gap error, closing speed) sets throttle
and brake. Steering stays with the driver.

Click the camera window, then:
  A    toggle ACC
  +/-  cruise speed
  Q    quit
"""

import time

import cv2
import numpy as np
from beamngpy.sensors import Electrics

from sim import open_beamng, wait_for_player_vehicle, front_camera, wait_for_frame
from vehicle_detection import VehicleDetector

DEPTH_FAR = 80.0

DIST_EMERGENCY = 5.0    # full brake below this
DIST_TARGET    = 12.0   # desired gap
DIST_LOST      = 70.0   # beyond this, treat the road as clear

CRUISE_SPEED_KMH = 100.0
SPEED_STEP       = 5.0
MAX_THROTTLE     = 0.70
MAX_BRAKE        = 0.85

FOLLOW_KP = 0.08   # gap error gain
FOLLOW_KD = 0.25   # closing speed gain
CRUISE_KP = 0.03

CTRL_INTERVAL = 0.05


def acc_control(dist_m, speed_kmh, cruise_kmh, rel_speed_ms):
    """Returns (throttle, brake)."""
    if dist_m is None or dist_m > DIST_LOST:
        err = cruise_kmh - speed_kmh
        if err > 0:
            return float(np.clip(err * CRUISE_KP, 0.0, MAX_THROTTLE)), 0.0
        return 0.0, float(np.clip(-err * 0.05, 0.0, MAX_BRAKE))

    if dist_m < DIST_EMERGENCY:
        return 0.0, MAX_BRAKE

    # rel_speed_ms < 0 means we are closing in, which pulls the output towards braking
    pd = (dist_m - DIST_TARGET) * FOLLOW_KP + rel_speed_ms * FOLLOW_KD
    if pd > 0:
        if speed_kmh >= cruise_kmh:
            return 0.0, 0.0
        return float(np.clip(pd, 0.0, MAX_THROTTLE)), 0.0
    return 0.0, float(np.clip(-pd, 0.0, MAX_BRAKE))


def main():
    global CRUISE_SPEED_KMH

    detector = VehicleDetector(0.05, DEPTH_FAR, DIST_EMERGENCY, DIST_TARGET)
    bng = open_beamng()
    vehicle = wait_for_player_vehicle(bng)
    electrics = Electrics()
    vehicle.sensors.attach('electrics', electrics)

    cam_colour = front_camera(bng, vehicle, 'acc_colour')
    cam_depth  = front_camera(bng, vehicle, 'acc_depth', colour=False, depth=True, depth_far=DEPTH_FAR)
    wait_for_frame(cam_colour)
    print("Ready. Click the camera window: A ACC, +/- speed, Q quit")

    active      = False
    prev_ctrl_t = time.time()
    prev_time   = time.time()
    prev_dist_m = None

    while True:
        colour_data = cam_colour.poll().get('colour')
        depth_data  = cam_depth.poll().get('depth')
        if colour_data is None:
            cv2.waitKey(1)
            continue
        frame = cv2.cvtColor(np.array(colour_data), cv2.COLOR_RGBA2BGR)
        display = frame.copy()

        dist_m = detector.detect(frame, depth_data, display)

        vehicle.sensors.poll()
        speed_kmh = float(electrics.data.get('airspeed', 0)) * 3.6

        now = time.time()
        dt = max(now - prev_time, 1e-3)
        rel_speed_ms = 0.0
        if dist_m is not None and prev_dist_m is not None and dist_m <= DIST_LOST:
            rel_speed_ms = (dist_m - prev_dist_m) / dt

        if active and (now - prev_ctrl_t) >= CTRL_INTERVAL:
            throttle, brake = acc_control(dist_m, speed_kmh, CRUISE_SPEED_KMH, rel_speed_ms)
            vehicle.control(throttle=throttle, brake=brake)
            prev_ctrl_t = now

        prev_dist_m = dist_m if (dist_m is not None and dist_m <= DIST_LOST) else None
        prev_time = now

        cv2.putText(display, f"ACC: {'ON' if active else 'OFF'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0) if active else (0, 0, 255), 2)
        cv2.putText(display, f"Speed:  {speed_kmh:5.1f} km/h", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"Target: {CRUISE_SPEED_KMH:5.1f} km/h", (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

        if dist_m is not None and dist_m <= DIST_LOST:
            gap_err = dist_m - DIST_TARGET
            if dist_m < DIST_EMERGENCY:
                label, col = f"EMERGENCY BRAKE  {dist_m:.1f}m", (0, 0, 255)
            elif gap_err < -2:
                label, col = f"TOO CLOSE  {dist_m:.1f}m", (0, 80, 255)
            elif abs(gap_err) <= 2:
                label, col = f"ON TARGET  {dist_m:.1f}m", (0, 255, 0)
            else:
                label, col = f"FOLLOWING  {dist_m:.1f}m", (0, 200, 255)
        else:
            label, col = "CLEAR", (180, 180, 180)
        cv2.putText(display, label, (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

        cv2.imshow('ACC', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            active = not active
            if not active:
                vehicle.control(throttle=0.0, brake=0.0)
            print(f"ACC {'on' if active else 'off'}")
        elif key in (ord('+'), ord('=')):
            CRUISE_SPEED_KMH = min(CRUISE_SPEED_KMH + SPEED_STEP, 130.0)
            print(f"Cruise speed: {CRUISE_SPEED_KMH:.0f} km/h")
        elif key == ord('-'):
            CRUISE_SPEED_KMH = max(CRUISE_SPEED_KMH - SPEED_STEP, 10.0)
            print(f"Cruise speed: {CRUISE_SPEED_KMH:.0f} km/h")

    cv2.destroyAllWindows()
    cam_colour.remove()
    cam_depth.remove()
    vehicle.control(throttle=0.0, brake=0.5)
    bng.disconnect()


if __name__ == '__main__':
    main()
