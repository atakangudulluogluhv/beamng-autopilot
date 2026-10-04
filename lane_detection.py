"""
Lane detection with classic computer vision:
HLS colour threshold -> Canny -> trapezoid ROI -> Hough lines -> left/right lane fit.

Each side is fitted as x = m*y + b (x as a function of y, so near-vertical lines
are well behaved) and smoothed with a median over the last few frames.
"""

from collections import deque

import cv2
import numpy as np

# Trapezoid region of interest, as fractions of the frame
ROI_BOT_L, ROI_BOT_R = 0.10, 0.90
ROI_TOP_L, ROI_TOP_R = 0.33, 0.67
ROI_TOP_Y            = 0.52   # low enough to cut out sky and buildings

HOUGH_THRESHOLD    = 25
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 80

SMOOTH_FRAMES = 10


def isolate_lane_markings(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white  = cv2.inRange(hls, np.array([0, 190, 0]),  np.array([180, 255, 255]))   # low L to catch faded paint
    yellow = cv2.inRange(hls, np.array([15, 80, 80]), np.array([40, 255, 255]))
    # Colour mask only. Running edges on the raw image picks up walls and buildings.
    mask  = cv2.bitwise_or(white, yellow)
    edges = cv2.Canny(cv2.GaussianBlur(mask, (5, 5), 0), 80, 160)
    return edges, mask


def region_of_interest(img, h, w):
    mask = np.zeros_like(img)
    poly = np.array([[
        (int(w * ROI_BOT_L), h),
        (int(w * ROI_TOP_L), int(h * ROI_TOP_Y)),
        (int(w * ROI_TOP_R), int(h * ROI_TOP_Y)),
        (int(w * ROI_BOT_R), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask), poly


class LaneDetector:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self._left_history  = deque(maxlen=SMOOTH_FRAMES)
        self._right_history = deque(maxlen=SMOOTH_FRAMES)
        self._virtual_cx    = float(w // 2)
        self.confidence     = 0.0
        self.lane_width_px  = 0

    @staticmethod
    def _median_line(segments):
        ms, bs = [], []
        for (x1, y1, x2, y2) in segments:
            if y2 == y1:
                continue
            m = (x2 - x1) / (y2 - y1)
            ms.append(m)
            bs.append(x1 - m * y1)
        if not ms:
            return None
        return float(np.median(ms)), float(np.median(bs))

    @staticmethod
    def _coords(params, y_bot, y_top):
        if params is None:
            return None
        m, b = params
        return (int(m * y_bot + b), y_bot, int(m * y_top + b), y_top)

    def _sane(self, coords, side):
        if coords is None:
            return False
        x_bot = coords[0]
        if side == 'L' and x_bot > self.w * 0.55:
            return False
        if side == 'R' and x_bot < self.w * 0.45:
            return False
        return 5 < x_bot < self.w - 5

    def detect(self, frame):
        """
        Returns (error_px, confidence, annotated, colour_mask, roi_edges).
        error_px is lane centre minus image centre, or None when confidence is too low.
        """
        h, w = self.h, self.w
        cx = w // 2
        y_bot, y_top = h, int(h * ROI_TOP_Y)

        edges, colour_mask = isolate_lane_markings(frame)
        roi, poly = region_of_interest(edges, h, w)
        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, HOUGH_THRESHOLD,
                                minLineLength=HOUGH_MIN_LINE_LEN,
                                maxLineGap=HOUGH_MAX_LINE_GAP)

        # Split segments by slope and side of the image
        l_lines, r_lines = [], []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                if abs(y2 - y1) < 10:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                mid_x = (x1 + x2) / 2
                if -1.5 < slope < -0.25 and mid_x < cx:
                    l_lines.append((x1, y1, x2, y2))
                elif 0.25 < slope < 1.5 and mid_x > cx:
                    r_lines.append((x1, y1, x2, y2))

        l_params = self._median_line(l_lines)
        r_params = self._median_line(r_lines)
        l_cand = self._coords(l_params, y_bot, y_top)
        r_cand = self._coords(r_params, y_bot, y_top)

        # Only accept a pair if the lane width is plausible
        if l_params and self._sane(l_cand, 'L'):
            if r_cand and self._sane(r_cand, 'R'):
                if w * 0.20 < abs(r_cand[0] - l_cand[0]) < w * 0.85:
                    self._left_history.append(l_params)
                    self._right_history.append(r_params)
            else:
                self._left_history.append(l_params)
        elif r_params and self._sane(r_cand, 'R'):
            self._right_history.append(r_params)

        l_avg = np.median(self._left_history, axis=0) if self._left_history else None
        r_avg = np.median(self._right_history, axis=0) if self._right_history else None
        left  = self._coords(l_avg, y_bot, y_top)
        right = self._coords(r_avg, y_bot, y_top)
        if not self._sane(left, 'L'):
            left = None
        if not self._sane(right, 'R'):
            right = None

        detected_cx = None
        if left and right:
            width = abs(right[0] - left[0])
            ok = w * 0.20 < width < w * 0.85
            if self.lane_width_px > 0:
                ok = ok and abs(width - self.lane_width_px) / self.lane_width_px < 0.30
            if ok:
                self.lane_width_px = width if self.lane_width_px == 0 else \
                                     int(0.92 * self.lane_width_px + 0.08 * width)
                detected_cx = (left[2] + right[2]) // 2
                self.confidence = min(1.0, self.confidence + 0.15)
            else:
                self.confidence = max(0.0, self.confidence - 0.10)
        elif left:
            # One line only: estimate the centre from the last known lane width
            detected_cx = left[2] + self.lane_width_px // 2
            self.confidence = min(0.65, self.confidence + 0.04)
        elif right:
            detected_cx = right[2] - self.lane_width_px // 2
            self.confidence = min(0.65, self.confidence + 0.04)
        else:
            self.confidence = max(0.0, self.confidence - 0.10)

        # Move the tracked centre at most 15 px per frame
        if detected_cx is not None:
            detected_cx = int(np.clip(detected_cx, w * 0.15, w * 0.85))
            self._virtual_cx += np.clip(detected_cx - self._virtual_cx, -15, 15)

        annotated = frame.copy()
        cv2.polylines(annotated, poly, True, (0, 255, 255), 1)
        if left:
            cv2.line(annotated, left[:2], left[2:], (255, 0, 0), 3)
        if right:
            cv2.line(annotated, right[:2], right[2:], (0, 0, 255), 3)
        lane_cx = int(self._virtual_cx)
        cv2.line(annotated, (lane_cx, h - 10), (lane_cx, h - 60), (0, 255, 0), 2)
        cv2.line(annotated, (cx, h - 10),      (cx, h - 60),      (255, 255, 0), 2)

        error = (lane_cx - cx) if self.confidence > 0.25 else None
        return error, self.confidence, annotated, colour_mask, roi
