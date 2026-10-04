"""
Autopilot: lane keeping + adaptive cruise control.

Steering combines two sources:
  - geometric lane detection (lane_detection.py) through a PID controller
  - the DAVE-2 behavioral cloning network (dave2.py)
They are blended by lane detection confidence. When the lane markings are
clear the PID does most of the work; when they fade or disappear the network
takes over.

Speed is set by:
  - ACC: YOLOv8 + depth camera find the lead car, a PD controller keeps the gap
  - curve slowdown: the target speed drops as the steering angle grows
Steering, throttle and brake are all rate limited so the car doesn't jerk.

Usage: start BeamNG.tech, get in a car in freeroam, run this script,
click the camera window, then:
  A    toggle autopilot
  +/-  cruise speed
  D    debug view (lane mask)
  Q    quit
"""

import os
import time

import cv2
import numpy as np
import torch
from beamngpy.sensors import Electrics

from config import CAM_WIDTH, CAM_HEIGHT, MODEL_PATH
from dave2 import load_model, preprocess
from lane_detection import LaneDetector
from sim import PID, open_beamng, wait_for_player_vehicle, front_camera, wait_for_frame
from vehicle_detection import VehicleDetector

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEPTH_FAR = 80.0

# Lateral
LANE_KP, LANE_KI, LANE_KD = 0.0044, 0.000018, 0.0000001   # pixel error -> steering
LANE_DEADZONE_PX = 8
MAX_STEER        = 0.45
MAX_STEER_RATE   = 0.030   # max steering change per control tick
STEER_DEADZONE   = 0.012
STEER_EMA        = 0.35
BC_MIN_WEIGHT    = 0.15    # the network always keeps at least this share

# ACC (metres)
DIST_EMERGENCY = 5.0
DIST_TARGET    = 12.0
DIST_LOST      = 70.0
FOLLOW_KP      = 0.10
FOLLOW_KD      = 0.30

# Speed
CRUISE_SPEED_KMH     = 45.0
SPEED_STEP           = 5.0
SPEED_MIN, SPEED_MAX = 10.0, 120.0
SPEED_KP             = 0.04
MAX_THROTTLE         = 0.65
MAX_BRAKE            = 0.85
THROTTLE_RATE        = 0.04
BRAKE_RATE           = 0.10

# Curve slowdown: at CURVE_FULL_STEER the target speed is cut by CURVE_SLOW_GAIN,
# but never below CURVE_MIN_FRAC of the cruise speed.
CURVE_SLOW_GAIN  = 0.85
CURVE_FULL_STEER = 0.30
CURVE_MIN_FRAC   = 0.35

CTRL_INTERVAL = 0.05

MAIN_WINDOW  = 'Autopilot'
DEBUG_WINDOW = 'Debug: lane mask | ROI edges'


def longitudinal_control(dist_m, speed_kmh, cruise_kmh, rel_speed_ms, steer_mag):
    """Returns (throttle, brake, target_speed_kmh)."""
    curve_frac = max(CURVE_MIN_FRAC, 1.0 - CURVE_SLOW_GAIN * min(1.0, steer_mag / CURVE_FULL_STEER))
    v_target = cruise_kmh * curve_frac

    # No lead car: plain cruise control
    if dist_m is None or dist_m > DIST_LOST:
        err = v_target - speed_kmh
        if err > 0:
            return float(np.clip(err * SPEED_KP, 0.0, MAX_THROTTLE)), 0.0, v_target
        return 0.0, float(np.clip(-err * 0.05, 0.0, MAX_BRAKE)), v_target

    if dist_m < DIST_EMERGENCY:
        return 0.0, MAX_BRAKE, v_target

    # Following: gap error + closing speed (negative rel_speed = closing in)
    pd = (dist_m - DIST_TARGET) * FOLLOW_KP + rel_speed_ms * FOLLOW_KD
    if pd > 0:
        if speed_kmh >= v_target:
            return 0.0, 0.0, v_target
        return float(np.clip(pd, 0.0, MAX_THROTTLE)), 0.0, v_target
    return 0.0, float(np.clip(-pd, 0.0, MAX_BRAKE)), v_target


def draw_hud(display, autopilot, speed_kmh, v_target, steer, w_lane, lane_conf,
             throttle, brake, dist_m):
    ap_col = (0, 255, 0) if autopilot else (0, 0, 255)
    cv2.putText(display, f"AUTOPILOT: {'ON' if autopilot else 'OFF'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, ap_col, 2)
    cv2.putText(display, f"Speed {speed_kmh:5.1f}  ->  Target {v_target:5.1f} km/h", (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Which steering source is in charge: green = lane PID, magenta = network
    src = "LANE" if w_lane > 0.6 else "BLEND" if w_lane > 0.2 else "BC-NET"
    src_col = (0, 255, 0) if w_lane > 0.6 else (0, 200, 255) if w_lane > 0.2 else (255, 0, 200)
    cv2.putText(display, f"Steer {steer:+.2f}  src={src} (lane {lane_conf:.0%})", (10, 88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, src_col, 2)
    cv2.putText(display, f"Throttle {throttle:.2f}  Brake {brake:.2f}", (10, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    if dist_m is not None and dist_m <= DIST_LOST:
        gap_err = dist_m - DIST_TARGET
        if dist_m < DIST_EMERGENCY:
            label, col = f"EMERGENCY BRAKE {dist_m:.1f}m", (0, 0, 255)
        elif gap_err < -2:
            label, col = f"TOO CLOSE {dist_m:.1f}m", (0, 80, 255)
        elif abs(gap_err) <= 2:
            label, col = f"ON TARGET {dist_m:.1f}m", (0, 255, 0)
        else:
            label, col = f"FOLLOWING {dist_m:.1f}m", (0, 200, 255)
    else:
        label, col = "ROAD CLEAR", (180, 180, 180)
    cv2.putText(display, label, (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # Steering bar
    bcx = CAM_WIDTH // 2
    cv2.rectangle(display, (bcx - 200, CAM_HEIGHT - 24), (bcx + 200, CAM_HEIGHT - 10), (50, 50, 50), -1)
    cv2.rectangle(display, (bcx, CAM_HEIGHT - 24), (int(bcx + steer / MAX_STEER * 200), CAM_HEIGHT - 10),
                  src_col, -1)
    cv2.line(display, (bcx, CAM_HEIGHT - 28), (bcx, CAM_HEIGHT - 6), (255, 255, 255), 1)


def main():
    global CRUISE_SPEED_KMH

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}. Train one with bc_train.py first.")
        return

    model = load_model(MODEL_PATH, DEVICE)
    print(f"BC model loaded ({DEVICE})")
    detector = VehicleDetector(0.05, DEPTH_FAR, DIST_EMERGENCY, DIST_TARGET)
    lane     = LaneDetector(CAM_WIDTH, CAM_HEIGHT)
    lane_pid = PID(LANE_KP, LANE_KI, LANE_KD)

    bng = open_beamng()
    vehicle = wait_for_player_vehicle(bng)
    electrics = Electrics()
    vehicle.sensors.attach('electrics', electrics)

    cam_colour = front_camera(bng, vehicle, 'ap_colour')
    cam_depth  = front_camera(bng, vehicle, 'ap_depth', colour=False, depth=True, depth_far=DEPTH_FAR)
    wait_for_frame(cam_colour)
    print("Ready. Click the camera window: A autopilot, +/- speed, D debug, Q quit")

    autopilot     = False
    debug_view    = False
    current_steer = 0.0
    smooth_steer  = 0.0
    throttle_cmd  = 0.0
    brake_cmd     = 0.0
    prev_ctrl_t   = time.time()
    prev_time     = time.time()
    prev_dist_m   = None

    while True:
        colour_data = cam_colour.poll().get('colour')
        depth_data  = cam_depth.poll().get('depth')
        if colour_data is None:
            cv2.waitKey(1)
            continue
        frame = cv2.cvtColor(np.array(colour_data), cv2.COLOR_RGBA2BGR)

        now = time.time()
        dt  = max(now - prev_time, 1e-3)

        # Perception
        lane_err, lane_conf, display, colour_mask, roi_edges = lane.detect(frame)
        dist_m = detector.detect(frame, depth_data, display)
        with torch.no_grad():
            bc_steer = float(model(preprocess(frame, DEVICE)).item())

        # Steering fusion
        if lane_err is not None:
            if abs(lane_err) < LANE_DEADZONE_PX:
                lane_err = 0.0
            lane_steer = float(np.clip(lane_pid.step(lane_err, dt), -MAX_STEER, MAX_STEER))
            w_lane = float(np.clip(lane_conf, 0.0, 1.0)) * (1.0 - BC_MIN_WEIGHT)
        else:
            lane_steer, w_lane = 0.0, 0.0
        fused_steer  = w_lane * lane_steer + (1.0 - w_lane) * bc_steer
        smooth_steer = STEER_EMA * fused_steer + (1.0 - STEER_EMA) * smooth_steer
        target_steer = 0.0 if abs(smooth_steer) < STEER_DEADZONE else \
                       float(np.clip(smooth_steer, -MAX_STEER, MAX_STEER))

        vehicle.sensors.poll()
        speed_kmh = float(electrics.data.get('airspeed', 0)) * 3.6
        rel_speed_ms = 0.0
        if dist_m is not None and prev_dist_m is not None and dist_m <= DIST_LOST:
            rel_speed_ms = (dist_m - prev_dist_m) / dt

        # Control, rate limited
        if autopilot and (now - prev_ctrl_t) >= CTRL_INTERVAL:
            current_steer += np.clip(target_steer - current_steer, -MAX_STEER_RATE, MAX_STEER_RATE)

            t_des, b_des, v_target = longitudinal_control(
                dist_m, speed_kmh, CRUISE_SPEED_KMH, rel_speed_ms, abs(current_steer))
            throttle_cmd += float(np.clip(t_des - throttle_cmd, -THROTTLE_RATE, THROTTLE_RATE))
            brake_cmd    += float(np.clip(b_des - brake_cmd, -BRAKE_RATE, BRAKE_RATE))
            throttle_cmd = float(np.clip(throttle_cmd, 0.0, MAX_THROTTLE))
            brake_cmd    = float(np.clip(brake_cmd, 0.0, MAX_BRAKE))

            vehicle.control(steering=float(current_steer), throttle=throttle_cmd, brake=brake_cmd)
            prev_ctrl_t = now
        else:
            v_target = CRUISE_SPEED_KMH

        prev_dist_m = dist_m if (dist_m is not None and dist_m <= DIST_LOST) else None
        prev_time   = now

        draw_hud(display, autopilot, speed_kmh, v_target, current_steer, w_lane, lane_conf,
                 throttle_cmd, brake_cmd, dist_m)
        cv2.imshow(MAIN_WINDOW, display)

        if debug_view:
            dbg = np.hstack([cv2.cvtColor(colour_mask, cv2.COLOR_GRAY2BGR),
                             cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)])
            cv2.imshow(DEBUG_WINDOW, cv2.resize(dbg, (CAM_WIDTH, CAM_HEIGHT // 2)))
        else:
            try:
                cv2.destroyWindow(DEBUG_WINDOW)
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            autopilot = not autopilot
            current_steer = smooth_steer = throttle_cmd = brake_cmd = 0.0
            lane_pid.reset()
            if not autopilot:
                vehicle.control(steering=0.0, throttle=0.0, brake=0.0)
            print(f"Autopilot {'on' if autopilot else 'off'}")
        elif key in (ord('+'), ord('=')):
            CRUISE_SPEED_KMH = min(CRUISE_SPEED_KMH + SPEED_STEP, SPEED_MAX)
            print(f"Cruise speed: {CRUISE_SPEED_KMH:.0f} km/h")
        elif key == ord('-'):
            CRUISE_SPEED_KMH = max(CRUISE_SPEED_KMH - SPEED_STEP, SPEED_MIN)
            print(f"Cruise speed: {CRUISE_SPEED_KMH:.0f} km/h")
        elif key == ord('d'):
            debug_view = not debug_view

    cv2.destroyAllWindows()
    cam_colour.remove()
    cam_depth.remove()
    vehicle.control(throttle=0.0, brake=0.5, steering=0.0)
    bng.disconnect()


if __name__ == '__main__':
    main()
