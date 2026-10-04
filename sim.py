"""Helpers for connecting to BeamNG.tech and attaching sensors."""

import time

from beamngpy import BeamNGpy
from beamngpy.sensors import Camera

from config import (BNG_HOME, BNG_BINARY, HOST, PORT,
                    CAM_WIDTH, CAM_HEIGHT, CAM_POS, CAM_DIR, CAM_FOV)


class PID:
    def __init__(self, kp, ki, kd, i_limit=5.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit = i_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0
        self.last_p = self.last_i = self.last_d = 0.0

    def step(self, error, dt):
        self._integral = float(min(max(self._integral + error * dt, -self.i_limit), self.i_limit))
        derivative = (error - self._prev_err) / max(dt, 1e-6)
        self._prev_err = error
        self.last_p = self.kp * error
        self.last_i = self.ki * self._integral
        self.last_d = self.kd * derivative
        return self.last_p + self.last_i + self.last_d


def open_beamng():
    """Connect to a running BeamNG.tech, or start one."""
    bng = BeamNGpy(HOST, PORT, home=BNG_HOME, binary=BNG_BINARY)
    bng.open(launch=True)
    return bng


def wait_for_player_vehicle(bng):
    """Block until the user has spawned into a vehicle, then connect to it."""
    print("Load freeroam and get in a car. Waiting", end="", flush=True)
    vid = None
    while vid is None:
        try:
            vid = bng.vehicles.get_player_vehicle_id()['vid']
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)
    vehicle = bng.vehicles.get_current(include_config=False)[vid]
    vehicle.connect(bng)
    print(f"\nConnected to vehicle {vid}")
    return vehicle


def front_camera(bng, vehicle, name, colour=True, depth=False, depth_far=80.0):
    kwargs = {}
    if depth:
        kwargs = dict(near_far_planes=(0.05, depth_far), integer_depth=True)
    return Camera(name=name, bng=bng, vehicle=vehicle,
                  pos=CAM_POS, dir=CAM_DIR, field_of_view_y=CAM_FOV,
                  resolution=(CAM_WIDTH, CAM_HEIGHT),
                  is_render_colours=colour, is_render_depth=depth,
                  is_render_annotations=False, requested_update_time=0.05,
                  **kwargs)


def wait_for_frame(camera):
    while camera.poll().get('colour') is None:
        time.sleep(0.05)
