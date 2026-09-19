import cv2
import numpy as np
import tensorflow as tf
import time

MODEL_PATH = "kaggle_mobilenet_model.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

class_names = [
    "fist", "five", "none", "okay",
    "peace", "rad", "straight", "thumbs"
]

gesture_scores = {name: 0 for name in class_names}
confirmed_gesture = "none"
MAX_SCORE = 6

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Webcam started. Press Q to exit.")
previous_time = time.time()

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_height, frame_width = frame.shape[:2]

    # 300 x 300 hand region
    box_size = 300
    x1 = (frame_width - box_size) // 2
    y1 = (frame_height - box_size) // 2 + 40
    x2 = x1 + box_size
    y2 = y1 + box_size

    roi = frame[y1:y2, x1:x2]
    confidence = 0.0
    raw_prediction = "none"

    if roi.size > 0:
        # Original model preprocessing 
        resized_roi = cv2.resize(roi, (96, 96))
        image_array = resized_roi.astype(np.float32) / 255.0
        image_tensor = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_tensor, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = float(predictions[predicted_index])

        if confidence > 0.65:
            raw_prediction = class_names[predicted_index]

        # Score-based smoothing
        for name in class_names:
            if name == raw_prediction and raw_prediction != "none":
                gesture_scores[name] = min(gesture_scores[name] + 2, MAX_SCORE)
            else:
                gesture_scores[name] = max(gesture_scores[name] - 1, 0)

        highest_gesture = max(gesture_scores, key=gesture_scores.get)

        if gesture_scores[highest_gesture] >= 4:
            confirmed_gesture = highest_gesture
        elif gesture_scores[confirmed_gesture] == 0:
            confirmed_gesture = "none"

    # FPS
    current_time = time.time()
    fps = 1.0 / max(current_time - previous_time, 0.001)
    previous_time = current_time

    # Compact information panel
    panel_x1, panel_y1 = 15, 5
    panel_x2, panel_y2 = 420, 125

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (0, 0, 0),
        -1
    )
    frame = cv2.addWeighted(overlay, 0.58, frame, 0.42, 0)

    cv2.putText(
        frame,
        "HAND GESTURE RECOGNITION",
        (30, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )

    if confirmed_gesture == "none":
        shown_gesture = "Waiting for hand..."
        gesture_color = (255, 255, 255)
    else:
        shown_gesture = confirmed_gesture.upper()
        gesture_color = (0, 255, 0)

    cv2.putText(
        frame,
        f"Gesture : {shown_gesture}",
        (30, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        gesture_color,
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence * 100:.1f}%",
        (30, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"FPS : {fps:.1f}",
        (300, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        2
    )

    # Confidence bar
    bar_x, bar_y = 30, 111
    bar_width, bar_height = 350, 4

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + bar_width, bar_y + bar_height),
        (120, 120, 120),
        1
    )

    filled_width = int(bar_width * min(max(confidence, 0.0), 1.0))

    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + filled_width, bar_y + bar_height),
        (0, 255, 0),
        -1
    )

    # ROI instruction and box
    instruction = "Place your hand inside the box"
    text_width = cv2.getTextSize(
        instruction,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        1
    )[0][0]

    instruction_x = x1 + (box_size - text_width) // 2
    instruction_y = max(y1 - 12, panel_y2 + 22)

    cv2.putText(
        frame,
        instruction,
        (instruction_x, instruction_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 255, 0),
        1
    )

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Real-Time Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Application closed successfully.")
