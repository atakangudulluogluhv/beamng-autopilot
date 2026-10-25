"""
Summarizes CSV logs written by `autopilot.py --log`.

Usage:
  python log_summary.py data/autopilot_log_*.csv
"""

import argparse
import csv

import numpy as np


def summarize(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2:
        return None

    def col(name):
        return np.array([float(r[name]) if r[name] != '' else np.nan for r in rows])

    t, x, y = col('t'), col('x'), col('y')
    speed = col('speed_kmh')
    steer, w_lane = col('steer'), col('w_lane')
    lane_err, dist_m = col('lane_err'), col('dist_m')

    distance = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    duration = float(t[-1] - t[0])
    moving = speed > 5
    return {
        'file':              path,
        'duration_s':        duration,
        'distance_m':        distance,
        'mean_speed_kmh':    distance / duration * 3.6,
        'max_speed_kmh':     float(np.max(speed)),
        'control_hz':        float(1 / np.median(np.diff(t))),
        'lane_found_pct':    float(np.mean(~np.isnan(lane_err[moving])) * 100),
        'lane_weight_pct':   float(np.mean(w_lane[moving]) * 100),
        'steer_rate_rms':    float(np.sqrt(np.mean(np.diff(steer) ** 2))),
        'lead_car_pct':      float(np.mean(~np.isnan(dist_m)) * 100),
        'min_gap_m':         float(np.nanmin(dist_m)) if np.any(~np.isnan(dist_m)) else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()

    for path in args.logs:
        s = summarize(path)
        if s is None:
            print(f"{path}: empty")
            continue
        print(s['file'])
        print(f"  {s['duration_s']:.0f} s, {s['distance_m']:.0f} m, "
              f"mean {s['mean_speed_kmh']:.1f} km/h, max {s['max_speed_kmh']:.1f} km/h")
        print(f"  control loop {s['control_hz']:.1f} Hz, steering change per step (RMS) {s['steer_rate_rms']:.4f}")
        print(f"  while moving: lane found {s['lane_found_pct']:.0f}% of steps, "
              f"lane PID share of steering {s['lane_weight_pct']:.0f}%")
        gap = f", closest {s['min_gap_m']:.1f} m" if s['min_gap_m'] is not None else ''
        print(f"  lead car tracked {s['lead_car_pct']:.0f}% of steps{gap}")


if __name__ == '__main__':
    main()
