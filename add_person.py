import cv2
import os
import face_recognition
import time

DATASET_DIR = "dataset"
SAVE_INTERVAL = 0.4  # seconds between saves — gives enough variety without duplicates
MIN_IMAGES = 30


def capture_faces(name: str):
    save_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    existing = len(os.listdir(save_dir))
    if existing > 0:
        print(f"[INFO] {existing} images already exist for '{name}'. Adding more.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved = 0
    last_save = time.time()

    print(f"[INFO] Look at the camera. Need {MIN_IMAGES} clear face shots.")
    print("[INFO] Move your head slightly for variety. Press 'q' to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")

        if len(locations) == 1:
            top, right, bottom, left = locations[0]
            cv2.rectangle(display, (left, top), (right, bottom), (0, 220, 0), 2)

            now = time.time()
            if now - last_save >= SAVE_INTERVAL:
                img_path = os.path.join(save_dir, f"{name}_{existing + saved:04d}.jpg")
                cv2.imwrite(img_path, frame)
                saved += 1
                last_save = now

            status = f"Captured: {saved}/{MIN_IMAGES}"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        elif len(locations) > 1:
            cv2.putText(display, "Multiple faces — step back", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        else:
            cv2.putText(display, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow(f"Adding: {name}  |  q = done", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if saved >= MIN_IMAGES:
            print(f"\n[INFO] Got {MIN_IMAGES} images. You can press 'q' or wait.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Saved {saved} new images for '{name}' → {save_dir}")
    if saved < 10:
        print("[WARN] Less than 10 images captured — accuracy may be low. Run again.")


if __name__ == "__main__":
    name = input("Enter person's name: ").strip().replace(" ", "_")
    if not name:
        print("[ERROR] Name cannot be empty.")
    else:
        capture_faces(name)
