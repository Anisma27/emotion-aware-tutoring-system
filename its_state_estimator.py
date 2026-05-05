"""
ITS_State_Estimator — Emotion Inference Script
================================================
Project  : ITS_State_Estimator
Model    : emotion_model.h5  (Keras Sequential, 355,849 params)
Input    : 48x48 grayscale images
Output   : 7-class softmax (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
Platform : Linux / Ubuntu (CPU inference, no GPU required)

Usage
-----
  # Single image
  python its_state_estimator.py --image path/to/face.jpg

  # Folder of images
  python its_state_estimator.py --folder path/to/images/

  # Live webcam
  python its_state_estimator.py --webcam

  # Import as module
  from its_state_estimator import ITS_StateEstimator
  estimator = ITS_StateEstimator("emotion_model.h5")
  result = estimator.predict("face.jpg")
"""

import os
import sys
import argparse
import numpy as np

# Suppress TensorFlow info/warning logs (keep errors only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

INPUT_SIZE   = (48, 48)   # model expects 48x48
INPUT_SHAPE  = (48, 48, 1)  # (H, W, C) — single grayscale channel
NUM_CLASSES  = 7


# ── Core class ────────────────────────────────────────────────────────────────

class ITS_StateEstimator:
    """
    Loads emotion_model.h5 and provides inference methods for single images,
    batches, folder scans, and live webcam capture.

    Architecture (verified against saved weights):
        Conv2D(32) → MaxPool → Conv2D(64) → MaxPool →
        Conv2D(128) → MaxPool → Flatten →
        Dense(128, ReLU) → Dropout(0.5) → Dense(7, Softmax)
    """

    def __init__(self, model_path: str = "emotion_model.h5"):
        """
        Parameters
        ----------
        model_path : str
            Path to the .h5 Keras model file.
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'\n"
                "Make sure emotion_model.h5 is in the same directory as this script."
            )

        print(f"[ITS_StateEstimator] Loading model from '{model_path}' ...")
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Verify the model matches the expected spec
        assert self.model.input_shape  == (None, 48, 48, 1), \
            f"Unexpected input shape: {self.model.input_shape}"
        assert self.model.output_shape == (None, 7), \
            f"Unexpected output shape: {self.model.output_shape}"

        print(f"[ITS_StateEstimator] Model loaded. "
              f"Parameters: {self.model.count_params():,}")


    # ── Preprocessing ──────────────────────────────────────────────────────

    @staticmethod
    def preprocess_image(image_input) -> np.ndarray:
        """
        Converts an image (file path, PIL Image, or numpy array) into a
        model-ready tensor of shape (1, 48, 48, 1) with values in [0, 1].

        Steps
        -----
        1. Load / accept image
        2. Convert to grayscale (L mode)
        3. Resize to 48×48 using LANCZOS resampling
        4. Normalize pixel values to [0.0, 1.0]
        5. Reshape to (1, 48, 48, 1)  ← batch dimension + channel dimension
        """
        # --- Load ---
        if isinstance(image_input, str):
            if not os.path.isfile(image_input):
                raise FileNotFoundError(f"Image not found: '{image_input}'")
            img = Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError(
                f"image_input must be a file path, numpy array, or PIL Image. "
                f"Got: {type(image_input)}"
            )

        # --- Grayscale ---
        img = img.convert("L")  # "L" = 8-bit grayscale

        # --- Resize ---
        img = img.resize(INPUT_SIZE, Image.LANCZOS)

        # --- Normalize ---
        arr = np.array(img, dtype=np.float32) / 255.0  # shape: (48, 48)

        # --- Reshape for model: (1, 48, 48, 1) ---
        arr = arr.reshape(1, 48, 48, 1)

        return arr


    # ── Single-image inference ─────────────────────────────────────────────

    def predict(self, image_input, verbose: bool = True) -> dict:
        """
        Run inference on a single image.

        Parameters
        ----------
        image_input : str | np.ndarray | PIL.Image
            The input image (path, array, or PIL object).
        verbose : bool
            If True, print the top prediction to stdout.

        Returns
        -------
        dict with keys:
            predicted_emotion : str   — top emotion label
            confidence        : float — probability of top class (0–1)
            probabilities     : dict  — {label: probability} for all 7 classes
        """
        tensor = self.preprocess_image(image_input)

        # Model inference (Dropout is inactive at inference time)
        raw_output = self.model(tensor, training=False).numpy()[0]  # shape: (7,)

        # Map to labels
        probabilities = {
            label: float(prob)
            for label, prob in zip(EMOTION_LABELS, raw_output)
        }

        top_idx       = int(np.argmax(raw_output))
        top_emotion   = EMOTION_LABELS[top_idx]
        top_conf      = float(raw_output[top_idx])

        if verbose:
            print(f"\n  Predicted emotion : {top_emotion}")
            print(f"  Confidence        : {top_conf * 100:.1f}%")
            print("  All probabilities :")
            for label, prob in sorted(probabilities.items(),
                                      key=lambda x: -x[1]):
                bar = "█" * int(prob * 20)
                print(f"    {label:<10} {prob * 100:5.1f}%  {bar}")

        return {
            "predicted_emotion": top_emotion,
            "confidence":        top_conf,
            "probabilities":     probabilities,
        }


    # ── Batch inference ────────────────────────────────────────────────────

    def predict_batch(self, image_inputs: list) -> list:
        """
        Run inference on a list of images efficiently using a single
        batched forward pass.

        Parameters
        ----------
        image_inputs : list of str | np.ndarray | PIL.Image

        Returns
        -------
        list of result dicts (same format as predict())
        """
        tensors = np.vstack([
            self.preprocess_image(img) for img in image_inputs
        ])  # shape: (N, 48, 48, 1)

        raw_outputs = self.model(tensors, training=False).numpy()  # (N, 7)

        results = []
        for raw in raw_outputs:
            top_idx = int(np.argmax(raw))
            results.append({
                "predicted_emotion": EMOTION_LABELS[top_idx],
                "confidence":        float(raw[top_idx]),
                "probabilities": {
                    label: float(prob)
                    for label, prob in zip(EMOTION_LABELS, raw)
                },
            })
        return results


    # ── Folder scan ────────────────────────────────────────────────────────

    def predict_folder(self, folder_path: str) -> list:
        """
        Predict emotion for every image file in a folder.

        Supported extensions: .jpg, .jpeg, .png, .bmp, .tiff, .webp

        Parameters
        ----------
        folder_path : str

        Returns
        -------
        list of dicts, each with 'filename' + prediction keys
        """
        SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Not a directory: '{folder_path}'")

        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED
        ])

        if not image_files:
            print(f"[ITS_StateEstimator] No supported images found in '{folder_path}'")
            return []

        print(f"[ITS_StateEstimator] Found {len(image_files)} image(s) in '{folder_path}'")

        results = []
        for filename in image_files:
            path   = os.path.join(folder_path, filename)
            result = self.predict(path, verbose=False)
            result["filename"] = filename
            emotion = result["predicted_emotion"]
            conf    = result["confidence"] * 100
            print(f"  {filename:<40} → {emotion:<10} ({conf:.1f}%)")
            results.append(result)

        return results


    # ── Webcam (live) inference ────────────────────────────────────────────

    def run_webcam(self, camera_index: int = 0):
        """
        Opens the webcam and runs real-time emotion detection.
        Requires OpenCV (cv2). Install with: pip install opencv-python

        Controls
        --------
        Q  — quit
        S  — save current frame as 'snapshot.png'
        """
        try:
            import cv2
        except ImportError:
            print("[ERROR] OpenCV is required for webcam mode.")
            print("        Install it with: pip install opencv-python")
            sys.exit(1)

        # Load Haar Cascade face detector (ships with OpenCV)
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera index {camera_index}")
            sys.exit(1)

        print("[ITS_StateEstimator] Webcam started. Press Q to quit, S to save snapshot.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            for (x, y, w, h) in faces:
                # Crop face region
                face_roi = gray_frame[y:y + h, x:x + w]

                # Resize to 48x48
                face_pil = Image.fromarray(face_roi).resize(INPUT_SIZE, Image.LANCZOS)
                tensor   = np.array(face_pil, dtype=np.float32) / 255.0
                tensor   = tensor.reshape(1, 48, 48, 1)

                # Infer
                raw    = self.model(tensor, training=False).numpy()[0]
                top_idx = int(np.argmax(raw))
                emotion = EMOTION_LABELS[top_idx]
                conf    = raw[top_idx] * 100

                # Draw bounding box + label
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label_text = f"{emotion} ({conf:.1f}%)"
                cv2.putText(
                    frame, label_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2,
                )

                # Draw probability bars for top 3 emotions
                sorted_probs = sorted(
                    zip(EMOTION_LABELS, raw), key=lambda x: -x[1]
                )[:3]
                for i, (lbl, prob) in enumerate(sorted_probs):
                    bar_x, bar_y = 10, 30 + i * 28
                    cv2.rectangle(
                        frame,
                        (bar_x, bar_y),
                        (bar_x + int(prob * 150), bar_y + 18),
                        (0, 200, 200), -1,
                    )
                    cv2.putText(
                        frame, f"{lbl}: {prob * 100:.0f}%",
                        (bar_x + 4, bar_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1,
                    )

            cv2.imshow("ITS_StateEstimator — Live Emotion Detection", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imwrite("snapshot.png", frame)
                print("[ITS_StateEstimator] Snapshot saved as 'snapshot.png'")

        cap.release()
        cv2.destroyAllWindows()
        print("[ITS_StateEstimator] Webcam closed.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ITS_State_Estimator — Emotion detection using emotion_model.h5"
    )
    parser.add_argument(
        "--model", type=str, default="emotion_model.h5",
        help="Path to the .h5 model file (default: emotion_model.h5)"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=str,
        help="Path to a single image file for inference"
    )
    group.add_argument(
        "--folder", type=str,
        help="Path to a folder of images for batch inference"
    )
    group.add_argument(
        "--webcam", action="store_true",
        help="Run live webcam inference (requires opencv-python)"
    )

    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera index for webcam mode (default: 0)"
    )

    args = parser.parse_args()

    # Initialise estimator
    estimator = ITS_StateEstimator(model_path=args.model)

    if args.image:
        print(f"\n[ITS_StateEstimator] Running inference on: {args.image}")
        estimator.predict(args.image, verbose=True)

    elif args.folder:
        print(f"\n[ITS_StateEstimator] Scanning folder: {args.folder}")
        estimator.predict_folder(args.folder)

    elif args.webcam:
        estimator.run_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()