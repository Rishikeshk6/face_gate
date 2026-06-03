import os
import pickle
import numpy as np
import face_recognition
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_DIR = "dataset"
MODEL_PATH   = "models/svm_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


def load_image_encoding(args):
    """Load one image and return its 128-d face encoding. Returns None on failure."""
    image_path, person_name = args
    try:
        img = face_recognition.load_image_file(image_path)
        locations = face_recognition.face_locations(img, model="hog")
        if len(locations) != 1:          # skip if 0 or 2+ faces detected
            return None
        enc = face_recognition.face_encodings(img, known_face_locations=locations)[0]
        return (enc, person_name)
    except Exception:
        return None


def collect_image_paths():
    tasks = []
    for person in sorted(os.listdir(DATASET_DIR)):
        folder = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(folder):
            continue
        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for img_file in images:
            tasks.append((os.path.join(folder, img_file), person))
    return tasks


def train():
    os.makedirs("models", exist_ok=True)

    # ── 1. Discover images ────────────────────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset folder '{DATASET_DIR}' not found. Run add_person.py first.")
        return

    tasks = collect_image_paths()
    if not tasks:
        print("[ERROR] No images found in dataset/. Run add_person.py first.")
        return

    people = sorted({p for _, p in tasks})
    print(f"[INFO] Found {len(tasks)} images across {len(people)} people: {', '.join(people)}")

    # ── 2. Guard: SVM needs >=2 classes ──────────────────────────────────────
    if len(people) < 2:
        print(
            f"\n[ERROR] Only 1 person found ('{people[0]}').\n"
            "        SVM requires at least 2 people to train.\n"
            "        Run add_person.py to add another person, then re-run train.py."
        )
        return

    # ── 3. Extract encodings in parallel ──────────────────────────────────────
    print("[INFO] Extracting face encodings (parallel)...\n")
    encodings, labels = [], []
    done = skipped = 0

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = {pool.submit(load_image_encoding, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result:
                encodings.append(result[0])
                labels.append(result[1])
            else:
                skipped += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  Processed {done}/{len(tasks)} — kept {len(encodings)}, skipped {skipped}", end="\r")

    print(f"\n\n[INFO] Usable encodings: {len(encodings)}  |  Skipped: {skipped}")

    # ── 4. Guard: need enough samples ─────────────────────────────────────────
    if len(encodings) < 2:
        print("[ERROR] Need at least 2 valid face images to train.")
        return

    unique_labels_after = sorted(set(labels))
    if len(unique_labels_after) < 2:
        print(
            f"\n[ERROR] After encoding, only 1 person has usable images ('{unique_labels_after[0]}').\n"
            "        The other person's images were all skipped (no face / multiple faces detected).\n"
            "        Re-capture clearer images for the missing person and re-run train.py."
        )
        return

    # ── 5. Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    X  = np.array(encodings)

    # ── 6. Train / test split ─────────────────────────────────────────────────
    # Only stratify when every class has >=2 samples
    class_counts = np.bincount(y)
    can_split = len(X) >= 10 and np.all(class_counts >= 2)

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )
    else:
        print("[WARN] Not enough samples for a proper split — reporting training accuracy.")
        X_train, X_test, y_train, y_test = X, X, y, y

    # ── 7. Train SVM ──────────────────────────────────────────────────────────
    print("[INFO] Training SVM (linear kernel)...")
    clf = SVC(kernel="linear", probability=True, C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n--- Accuracy Report ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── 8. Save model + encoder ───────────────────────────────────────────────
    with open(MODEL_PATH,   "wb") as f: pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le,  f)

    print(f"[DONE] Model saved        → {MODEL_PATH}")
    print(f"[DONE] Label encoder saved → {ENCODER_PATH}")


if __name__ == "__main__":
    train()