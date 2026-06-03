# Face Gate — Real-time Face Recognition Access Control

A webcam-based face recognition system that grants or denies access using an SVM classifier trained on your own face data. Built with `face_recognition` (dlib under the hood) and OpenCV.

---

## How it works

1. **Capture** face images of each person using `add_person.py`
2. **Train** an SVM on those face embeddings using `train.py`
3. **Run** real-time recognition with `recognize.py` — it draws a green box + "Gate Open" for known faces above the confidence threshold, and red for unknowns

Detection runs in a background thread so the display stays smooth even on slower machines.

---

## Setup

### Prerequisites

- Python 3.8–3.11
- A webcam
- `cmake` and C++ build tools (needed by dlib)

**On Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y cmake build-essential libopenblas-dev liblapack-dev
```

**On macOS:**
```bash
brew install cmake
```

**On Windows:**  
Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the C++ workload, then install cmake from [cmake.org](https://cmake.org/download/).

---

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/face-gate.git
cd face-gate

# 2. Create a virtual environment
python -m venv myenv

# 3. Activate it
source myenv/bin/activate        # Linux / macOS
myenv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt
```

> `dlib` (pulled in by `face_recognition`) takes a few minutes to compile — that's normal.

---

## Usage

### Step 1 — Add a person
```bash
python add_person.py
```
Enter the person's name when prompted. Look at the camera and move your head slightly. The script saves ~30 images automatically, then waits for `q`.

Repeat for every person you want to register.

### Step 2 — Train the model
```bash
python train.py
```
This extracts 128-dimensional face embeddings and trains a linear SVM. Prints a classification report at the end. Saves the model to `models/`.

### Step 3 — Run the gate
```bash
python recognize.py
```
Real-time recognition starts. Green box = access granted, red = denied. Press `q` to quit.

---

## Project structure

```
face-gate/
├── add_person.py       # Capture face images for a new person
├── train.py            # Extract embeddings and train SVM
├── recognize.py        # Real-time recognition loop
├── requirements.txt
├── .gitignore
├── dataset/            # Created automatically — not tracked by git
│   └── alice/
│       ├── alice_0000.jpg
│       └── ...
└── models/             # Created after training — not tracked by git
    ├── svm_model.pkl
    └── label_encoder.pkl
```

---

## Configuration

At the top of `recognize.py`:

| Variable | Default | What it does |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `80.0` | Min % confidence to grant access |
| `SMOOTH_FRAMES` | `5` | Rolling window size for prediction smoothing |
| `SCALE` | `0.5` | Frame resize factor for detection speed |

Lowering `SCALE` to `0.25` makes detection faster on slow hardware. Raising `CONFIDENCE_THRESHOLD` reduces false positives.

---

## Pushing to GitHub

Dataset and models are excluded from git (`.gitignore`). Never commit face images — they're personal biometric data.

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/your-username/face-gate.git
git push -u origin main
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `face_recognition` | Face detection + 128-d embeddings via dlib |
| `opencv-python` | Webcam capture and display |
| `scikit-learn` | SVM training and label encoding |
| `numpy` | Array ops |

---

## License

MIT
