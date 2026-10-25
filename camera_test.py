"""
Quick setup check: connects to BeamNG.tech, spawns a car on west_coast_usa
if no scenario is running, grabs one camera frame and saves it as first_frame.png.
"""

import os

import cv2
import numpy as np

from config import ROOT, SPAWNS
from sim import open_beamng, start_scenario, front_camera, wait_for_frame


def main():
    bng = open_beamng()

    # get_current() raises on the Lua side when nothing is loaded
    try:
        running = bng.scenario.get_current() is not None
    except Exception:
        running = False

    if running:
        vehicle = list(bng.vehicles.get_current(include_config=False).values())[0]
        vehicle.connect(bng)
    else:
        vehicle = start_scenario(bng, *SPAWNS['west_coast_usa'])

    camera = front_camera(bng, vehicle, 'test_cam')
    wait_for_frame(camera)   # the first polls return None until the camera has rendered
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
