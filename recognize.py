import os
import pickle
import time
import numpy as np
import cv2
import face_recognition

MODEL_PATH   = "models/svm_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# ── Tunable settings ──────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60   # probability below this → "Unknown"
FRAME_SKIP           = 2      # process every Nth frame (1 = every frame)
SCALE_FACTOR         = 0.5    # resize frame before detection (speed vs accuracy)
CAMERA_INDEX         = 0      # change if you have multiple cameras


def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            "[ERROR] Model files not found.\n"
            "        Run train.py first to generate models/svm_model.pkl and models/label_encoder.pkl"
        )
    with open(MODEL_PATH,   "rb") as f: clf = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le  = pickle.load(f)
    return clf, le


def preprocess_frame(frame):
    """Resize + convert BGR→RGB for face_recognition."""
    small = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def predict_faces(rgb_small, clf, le):
    """
    Detect faces in rgb_small and return a list of:
      (top, right, bottom, left, name, confidence, is_known)
    All coordinates are already scaled back to original frame size.
    """
    locations = face_recognition.face_locations(rgb_small, model="hog")
    if not locations:
        return []

    encodings = face_recognition.face_encodings(rgb_small, known_face_locations=locations)
    results = []
    inv = 1.0 / SCALE_FACTOR

    for (top, right, bottom, left), enc in zip(locations, encodings):
        probs   = clf.predict_proba([enc])[0]
        best_i  = np.argmax(probs)
        conf    = probs[best_i]
        name    = le.classes_[best_i] if conf >= CONFIDENCE_THRESHOLD else "Unknown"
        is_known = conf >= CONFIDENCE_THRESHOLD

        # Scale coordinates back to original frame
        results.append((
            int(top * inv), int(right * inv),
            int(bottom * inv), int(left * inv),
            name, conf, is_known
        ))
    return results


def draw_results(frame, predictions):
    """Draw bounding boxes and labels on the original frame."""
    for (top, right, bottom, left, name, conf, is_known) in predictions:
        # Box colour: green = known, red = unknown
        colour = (0, 200, 0) if is_known else (0, 0, 220)

        # Bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), colour, 2)

        # Label background
        label      = f"{name}  {conf*100:.1f}%" if is_known else "Unknown"
        gate_text  = "Gate: OPEN" if is_known else "Gate: CLOSED"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)

        cv2.rectangle(frame, (left, bottom - th - 14), (left + tw + 8, bottom), colour, cv2.FILLED)
        cv2.putText(frame, label, (left + 4, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Gate status above box
        cv2.putText(frame, gate_text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    return frame


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 0), 2)
    return frame


def run():
    print("[INFO] Loading models...")
    try:
        clf, le = load_models()
    except FileNotFoundError as e:
        print(e)
        return

    known_people = list(le.classes_)
    print(f"[INFO] Recognising: {', '.join(known_people)}")
    print("[INFO] Press 'q' to quit.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        return

    # Try to set HD resolution; fall back gracefully
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count      = 0
    last_predictions = []   # cache predictions between skipped frames
    fps              = 0.0
    t_prev           = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame — retrying...")
            time.sleep(0.05)
            continue

        frame_count += 1
        should_infer = (frame_count % FRAME_SKIP == 0)

        if should_infer:
            rgb_small        = preprocess_frame(frame)
            last_predictions = predict_faces(rgb_small, clf, le)

            # FPS calculation
            t_now  = time.time()
            fps    = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev = t_now

        display = draw_results(frame.copy(), last_predictions)
        display = draw_fps(display, fps)

        cv2.imshow("Face Gate  |  press q to quit", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Bye!")


if __name__ == "__main__":
    run()