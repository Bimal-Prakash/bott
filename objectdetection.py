"""
hand_to_http_servo.py

Detect index-finger direction and send mapped servo angle (0-180) via HTTP GET
to the ESP32 web endpoint (example: http://10.42.89.86/servo?angle=90 or http://10.42.89.86/servo?90).

Dependencies:
    pip install mediapipe opencv-python requests
"""

import time
from collections import deque
import urllib.parse

import cv2
import numpy as np
import mediapipe as mp
import requests

# ---------- CONFIG ----------
TARGET_URL = "http://10.42.89.86/servo?"   # provided by you
MAP_360_TO_180 = True
INVERT_SERVO = False
SERVO_OFFSET = 0

SMOOTH_FRAMES = 6
DELTA_THRESHOLD = 1.5      # degrees (servo degrees) before sending
MAX_SEND_HZ = 15.0         # throttle HTTP requests
ARROW_LENGTH = 80
REQUEST_TIMEOUT = 0.6      # seconds
# ----------------------------

min_send_interval = 1.0 / MAX_SEND_HZ

def send_angle_http(target_base_url, angle_int):
    """
    Send angle via HTTP GET. Tries ?angle=<n> first, falls back to ?<n>.
    Returns True if request succeeded (status 200-399), else False.
    """
    # Preferred: ?angle=<n>
    params = {"angle": str(angle_int)}
    try:
        resp = requests.get(target_base_url, params=params, timeout=REQUEST_TIMEOUT)
        if 200 <= resp.status_code < 400:
            return True
    except Exception:
        pass

    # Fallback: append raw query like ?90
    try:
        url = target_base_url + str(angle_int)
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if 200 <= resp.status_code < 400:
            return True
    except Exception:
        pass

    return False

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: initial frame capture failed")
        cap.release()
        return
    canvas = np.zeros_like(frame)

    angle_history = deque(maxlen=SMOOTH_FRAMES)
    points = deque(maxlen=1024)

    last_sent = None
    last_send_time = 0.0

    print("Running — press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed — exiting")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            display_angle = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

                    tip_x, tip_y = int(tip.x * width), int(tip.y * height)
                    pip_x, pip_y = int(pip.x * width), int(pip.y * height)

                    points.appendleft((tip_x, tip_y))

                    vec_x = tip_x - pip_x
                    vec_y = pip_y - tip_y  # flip y so positive is up

                    if vec_x == 0 and vec_y == 0:
                        continue

                    angle_rad = np.arctan2(vec_y, vec_x)
                    angle_deg = np.degrees(angle_rad)
                    angle_deg_norm = angle_deg % 360

                    angle_history.append(angle_deg_norm)
                    display_angle = sum(angle_history) / len(angle_history)

                    # draw arrow and tip circle
                    end_x = int(tip_x + ARROW_LENGTH * np.cos(np.radians(display_angle)))
                    end_y = int(tip_y - ARROW_LENGTH * np.sin(np.radians(display_angle)))
                    cv2.arrowedLine(frame, (tip_x, tip_y), (end_x, end_y), (0,0,255), 3, tipLength=0.3)
                    cv2.circle(frame, (tip_x, tip_y), 6, (255,0,0), -1)
                    break

            # draw path
            for i in range(1, len(points)):
                if points[i-1] is None or points[i] is None:
                    continue
                cv2.line(canvas, points[i-1], points[i], (0,255,0), 2)

            combined = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)

            if display_angle is not None and not np.isnan(display_angle):
                # Mapping to servo
                if MAP_360_TO_180:
                    servo_angle = int(display_angle / 2.0)
                else:
                    servo_angle = int(display_angle)

                if INVERT_SERVO:
                    servo_angle = 180 - servo_angle

                servo_angle = servo_angle + int(SERVO_OFFSET)
                servo_angle = max(0, min(180, servo_angle))

                cv2.putText(combined, f"Finger: {display_angle:.1f}\u00b0  Servo: {servo_angle}°",
                            (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(combined, "0=right 90=up 180=left 270=down", (10,75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

                now = time.time()
                should_send = True
                if now - last_send_time < min_send_interval:
                    should_send = False
                if last_sent is not None and abs(servo_angle - last_sent) < DELTA_THRESHOLD:
                    should_send = False

                if should_send:
                    success = send_angle_http(TARGET_URL, servo_angle)
                    if success:
                        last_sent = servo_angle
                        last_send_time = now
                        # debug:
                        # print("[http] sent", servo_angle)
                    else:
                        print("[http] send failed (timeout or connection issue)")

            cv2.imshow("Hand -> HTTP Servo", combined)
            cv2.imshow("Canvas", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Cleaning up...")
        hands.close()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
