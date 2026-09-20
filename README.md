# Real-Time Hand Gesture Recognition

A Python application that recognizes hand gestures from a webcam feed using a trained MobileNetV2 model.

## Development and Attribution

I trained the MobileNetV2-based classifier in Google Colab using a hand-gesture dataset sourced from Kaggle. I also built the real-time application, including webcam integration, image preprocessing, confidence filtering, gesture display, and smoothing logic.

- **Dataset:** [Gestures Hand](https://www.kaggle.com/datasets/kritanjalijain/gestures-hand) by Kaggle user `kritanjalijain`.

## Features

- Live webcam prediction
- Eight gesture classes: `fist`, `five`, `none`, `okay`, `peace`, `rad`, `straight`, and `thumbs`
- Confidence filtering to reduce uncertain predictions
- Score-based smoothing to reduce flickering labels
- A centered guide box for consistent hand placement

## Project files

| File | Purpose |
|------|---------|
| `realtime_prediction.py` | Runs webcam-based gesture recognition. |
| `realtime_prediction_simple_ui.py` | Same recognizer with a simpler on-screen panel. |
| `kaggle_mobilenet_model.keras` | Trained MobileNetV2 model. |
| `requirements.txt` | Python packages needed to run the project. |

## How it works

1. Each webcam frame is cropped to the guide box, resized to 96x96, and normalized.
2. The MobileNetV2 model outputs a probability for each of the 8 gesture classes.
3. Predictions below a 0.65 confidence threshold are treated as `none` to reduce false positives.
4. A score-based smoothing system tracks a running score per gesture (incrementing on repeated detections, decaying otherwise) so the displayed label only changes once a gesture has a clear, stable lead — this prevents flickering between labels frame to frame.

## Setup

1. Clone the repository and open its folder in a terminal.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

## Run

```powershell
python realtime_prediction.py
```

Keep your hand inside the green box. Press `q` to close the webcam window.

## Notes

- Ensure that your webcam is connected and available to the application.
- The model file is included because it is small enough for a standard GitHub repository.
- The `venv` folder is intentionally excluded from Git; each user should create their own environment.

## Known limitations

- Performance depends on your machine: running inference on every frame can cause lower FPS or noticeable lag on slower CPUs or machines without GPU acceleration for TensorFlow.
- Recognition accuracy is sensitive to lighting and background. Use a plain, uncluttered background with no distracting objects behind your hand; busy scenes or poor lighting may reduce accuracy.
- For best recognition, match the hand pose, orientation, distance, and framing used in the dataset images. For example, show a peace sign in the same upright hand position as the training examples; different angles or finger placement can lower confidence.
- Only one hand gesture is tracked at a time, inside the fixed guide box.
- Tested primarily on Windows with the package versions in `requirements.txt`; other OS/Python combinations may require adjustments (especially for `opencv-python` webcam access).

## License

This project's code is released under the [MIT License](LICENSE). The training dataset is subject to its own terms — see the attribution above.
