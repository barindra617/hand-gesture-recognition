import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import time

# 1. Load the trained MobileNetV2 model
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "kaggle_mobilenet_model.keras"
print("Loading model... please wait...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# 2. Gesture classes used by the trained model
class_names = ['fist', 'five', 'none', 'okay', 'peace', 'rad', 'straight', 'thumbs']

#  INSTANT BALANCED SMOOTHING
confirmed_gesture = "none"
gesture_scores = {cls: 0 for cls in class_names}  # Track scores for each gesture
MAX_SCORE = 6  # Caps how high a score can go (keeps it fast)
last_confidence = 0.0


def draw_dashboard(frame, gesture, confidence, fps):
    """Draw a compact status panel without covering the hand guide area."""
    _, width, _ = frame.shape
    green = (0, 255, 0)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_x, panel_y = 12, 12
    panel_width, panel_height = min(width - 24, 560), 100

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    cv2.putText(frame, "HAND GESTURE RECOGNITION", (panel_x + 16, panel_y + 22),
                font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Gesture: {gesture.upper()}", (panel_x + 16, panel_y + 50),
                font, 0.62, green, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (panel_x + 16, panel_y + 76),
                font, 0.52, white, 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + panel_width - 130, panel_y + 76),
                font, 0.52, white, 1, cv2.LINE_AA)

    bar_left, bar_top = panel_x + 16, panel_y + 86
    bar_right, bar_bottom = panel_x + panel_width - 16, panel_y + 96
    cv2.rectangle(frame, (bar_left, bar_top), (bar_right, bar_bottom), (140, 140, 140), 1)
    filled_right = int(bar_left + (bar_right - bar_left) * max(0.0, min(confidence, 1.0)))
    cv2.rectangle(frame, (bar_left, bar_top), (filled_right, bar_bottom), green, -1)


# 3. Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open the webcam. Check that it is connected and not in use.")

print("\nWebcam starting... Advanced stability system active.")
print("Press 'q' to exit safely.")
print("Tip: for best results, match the hand pose/orientation used in the training")
print("dataset, use a plain/uncluttered background, and keep lighting even.")

fps = 0.0
fps_frame_count = 0
fps_timer = time.perf_counter()

# Wrapped in try/finally so the webcam and windows always release cleanly,
# even if an unexpected error occurs mid-loop.
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fps_frame_count += 1
        now = time.perf_counter()
        elapsed = now - fps_timer
        if elapsed >= 0.5:
            fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_timer = now

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        dashboard_bottom = 12 + 100 + 14   # panel_y + panel_height + margin
        hint_zone_height = 26              # reserved strip at the bottom for the hint line
        available_height = h - dashboard_bottom - hint_zone_height

        box_size = min(300, w, max(available_height, 40))
        x1 = (w - box_size) // 2
        y1 = dashboard_bottom + max(0, (available_height - box_size) // 2)
        x2, y2 = x1 + box_size, y1 + box_size

        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            img_96x96 = cv2.resize(roi, (96, 96))
            img_array = np.array(img_96x96, dtype=np.float32) / 255.0
            img_tensor = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_tensor, verbose=0)
            best_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][best_class_idx]
            last_confidence = float(confidence)

            # Lowered confidence threshold to 0.65 to catch hand in room lighting
            if confidence > 0.65:
                raw_guess = class_names[best_class_idx]
            else:
                raw_guess = "none"

            #  SMOOTHING SCORE LOGIC
            for cls in class_names:
                if cls == raw_guess and raw_guess != "none":
                    # Build up confidence quickly
                    gesture_scores[cls] = min(gesture_scores[cls] + 2, MAX_SCORE)
                else:
                    # Decay other scores slowly (stops the flickering!)
                    gesture_scores[cls] = max(gesture_scores[cls] - 1, 0)

            # Get the gesture with the highest current score
            highest_scoring_gesture = max(gesture_scores, key=gesture_scores.get)

            # Only switch the text if the new gesture has a clear lead
            if gesture_scores[highest_scoring_gesture] >= 4:
                confirmed_gesture = highest_scoring_gesture
            elif gesture_scores[confirmed_gesture] == 0:
                confirmed_gesture = "none"
            # ------------------------------

        draw_dashboard(frame, confirmed_gesture, last_confidence, fps)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Match dataset hand pose | Plain background | Even lighting",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam window closed cleanly.")
