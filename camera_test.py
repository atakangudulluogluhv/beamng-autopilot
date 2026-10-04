"""
Quick setup check: connects to BeamNG.tech, spawns a car on west_coast_usa
if no scenario is running, grabs one camera frame and saves it as first_frame.png.
"""

import os

import cv2
import numpy as np
from beamngpy import Scenario, Vehicle

from config import ROOT
from sim import open_beamng, front_camera


def main():
    bng = open_beamng()

    # get_current() raises on the Lua side when nothing is loaded
    try:
        running = bng.scenario.get_current() is not None
    except Exception:
        running = False

    if running:
        vehicle = list(bng.vehicles.get_current_vehicles().values())[0]
        vehicle.connect(bng)
    else:
        scenario = Scenario('west_coast_usa', 'camera_test')
        vehicle = Vehicle('ego', model='etk800', license='CAM')
        scenario.add_vehicle(vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.3826834, 0.9238795))
        scenario.make(bng)
        bng.scenario.load(scenario)
        bng.scenario.start()

    camera = front_camera(bng, vehicle, 'test_cam')
    bng.control.pause()
    vehicle.sensors.poll()
    frame = cv2.cvtColor(np.array(camera.poll()['colour']), cv2.COLOR_RGBA2BGR)

    path = os.path.join(ROOT, 'first_frame.png')
    cv2.imwrite(path, frame)
    print(f"Saved {path} {frame.shape}")

    cv2.imshow('Camera test', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    camera.remove()
    bng.control.resume()
    bng.disconnect()


if __name__ == '__main__':
    main()
