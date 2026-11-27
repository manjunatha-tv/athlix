# ATHIX — AI-Powered Sports Technique Analysis

<p align="center">
  <strong>Understand movement. Analyse technique. Improve performance.</strong>
</p>

<p align="center">
  An experimental computer-vision platform for analysing cricket batting technique through video-based human pose estimation and biomechanical measurements.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Prototype-orange" alt="Project status">
  <img src="https://img.shields.io/badge/Focus-Cricket%20Batting-green" alt="Focus">
  <img src="https://img.shields.io/badge/AI-YOLO%20Pose-blue" alt="YOLO Pose">
  <img src="https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript-3178C6" alt="Frontend">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688" alt="Backend">
</p>

---

## Overview

**ATHIX** is an AI-assisted sports analysis project designed to help athletes understand and evaluate their movement patterns.

The project explores how computer vision and human pose estimation can be used to analyse sports techniques from ordinary video recordings.

Instead of relying entirely on visual observation, ATHIX extracts body landmarks, calculates joint angles and compares movement patterns against a saved reference technique.

The initial development focuses on **cricket batting**, beginning with the Straight Drive.

ATHIX is being developed as a prototype, with the long-term goal of making sports technique analysis more accessible, interpretable and useful for athletes and coaches.

## The Problem

Athletes often depend on coaches or manual video review to understand the technical details of their movements.

However:

- Analysing movement frame by frame can be time-consuming.
- Small differences in body positioning can be difficult to identify visually.
- Athletes may not have access to professional sports-analysis equipment.
- Comparing an athlete's technique against a reference demonstration is not always straightforward.

ATHIX explores a computer-vision-based approach to make these measurements easier to inspect.

## Key Features

### 1. Video-Based Analysis

Upload a cricket batting video and process it through the ATHIX analysis pipeline.

### 2. YOLO Pose Estimation

Uses an Ultralytics YOLO pose model to detect human body keypoints across video frames.

Detected landmarks are used to estimate body positioning and calculate joint angles.

### 3. Joint-Angle Calculation

Calculates selected biomechanical angles, including:

- Left and right elbow angles
- Left and right knee angles
- Left and right shoulder angles
- Left and right hip angles

### 4. Movement Data Processing

Extracts pose landmarks and joint-angle measurements, applies smoothing and generates movement-phase data.

### 5. Annotated Video Visualisation

Displays the original recording alongside a pose-annotated video showing detected body keypoints and skeletal connections.

### 6. Trainer Reference Comparison

Allows an analysed trainer demonstration to be saved as a reference for a batting technique.

Future player recordings can then be compared against the saved reference using extracted joint-angle patterns.

### 7. Experimental Similarity Measurement

ATHIX calculates an experimental joint-angle similarity percentage between the player's recording and the saved trainer reference.

The comparison currently uses eight joint-angle patterns and 100 normalised time samples.

**Important:** This percentage represents similarity to the selected reference recording. It is not a scientifically validated technique score or a professional coaching grade.

### 8. 3D Cricket Visualisation

Includes a 3D cricket batter model as part of the technique visualisation experience.

---

## How ATHIX Works

The current analysis workflow follows these stages:

```text
          CRICKET VIDEO
                |
                v
        VIDEO VALIDATION
                |
                v
        YOLO POSE ESTIMATION
                |
                v
        BODY LANDMARK EXTRACTION
                |
                v
        JOINT-ANGLE CALCULATION
                |
                v
         ANGLE SMOOTHING
                |
                v
       MOVEMENT-PHASE DETECTION
                |
                v
        REFERENCE COMPARISON
                |
                v
       ANALYSIS & VISUALISATION
```

### Stage 1 — Video Input

The user uploads a cricket batting video.

The application validates the recording before processing.

### Stage 2 — Pose Estimation

The YOLO pose model detects body landmarks in the video frames.

These landmarks represent important body positions such as the shoulders, elbows, wrists, hips, knees and ankles.

### Stage 3 — Joint-Angle Calculation

The extracted landmarks are used to calculate selected joint angles using geometric calculations.

### Stage 4 — Smoothing

The angle measurements are processed to reduce frame-to-frame fluctuations.

### Stage 5 — Movement-Phase Detection

The system analyses movement measurements to generate movement-phase labels.

Phase detection is currently experimental and requires further validation on different batting recordings.

### Stage 6 — Reference Comparison

The player's extracted joint-angle patterns are compared with a saved trainer reference for the selected technique.

### Stage 7 — Results

The frontend presents the analysis, annotated video, joint-angle measurements and experimental similarity results.

---

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React |
| Programming language | TypeScript |
| Frontend build tool | Vite |
| Backend | Python |
| Backend framework | FastAPI |
| Pose estimation | Ultralytics YOLO Pose |
| Computer vision | OpenCV |
| Numerical processing | NumPy |
| Tabular data processing | Pandas |
| 3D model asset | GLB |

---

## Project Structure(v0.1)

```text
ATHLIX/
│
├── backend/
│   ├── main.py
│   ├── analyze_straight_drive.py
│   ├── batting_config.py
│   ├── calculate_angles.py
│   ├── extract_landmarks.py
│   ├── phase_detection.py
│   ├── plot_smoothed_angles.py
│   ├── pose_test.py
│   ├── reference_skeleton.py
│   ├── smooth_angles.py
│   └── visualize_straight_drive.py
│
├── frontend/
│   ├── public/
│   │   └── models/
│   │       └── cricket-batter.glb
│   │
│   └── src/
│       ├── assets/
│       ├── components/
│       │   └── CricketScene.tsx
│       ├── App.tsx
│       ├── App.css
│       ├── index.css
│       └── main.tsx
│
├── .gitignore
├── package.json
└── package-lock.json
```

Generated videos, CSV files, private trainer references, model weights and virtual environments are excluded from the public repository.

---

## Getting Started

### Prerequisites

Make sure you have installed:

- Python 3.10 or a compatible version for the backend dependencies.
- Node.js and npm.
- Git.
- A computer capable of running the YOLO pose model.

### 1. Clone the Repository

```bash
git clone https://github.com/manjunatha-tv/athlix.git
cd athlix
```

### 2. Set Up the Backend

Open a terminal:

```bash
cd backend
```

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the backend packages:

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy pandas ultralytics
```

The exact dependency versions and a dedicated backend requirements file will be added as the project setup is stabilised.

Make sure the required YOLO pose model weights are available locally. Model weights are not included in this repository.

Start the backend:

```bash
python -m uvicorn main:app --reload
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

### 3. Set Up the Frontend

Open another terminal:

```bash
cd frontend
```

Install frontend dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed by Vite in your browser.

---

## Trainer Reference Workflow

One of the experimental features of ATHIX is reference-based technique comparison.

The intended workflow is:

1. Upload a trainer's demonstration video.
2. Analyse the video and extract its pose landmarks and joint angles.
3. Save the extracted measurements as a reference for a technique, such as Straight Drive.
4. Upload a player's recording.
5. Analyse the player's movement using the same pipeline.
6. Compare the player's measurements against the saved trainer reference.
7. View the experimental similarity results.

This approach uses previously extracted trainer measurements instead of requiring the trainer's video to be processed every time a player is analysed.

---

## Current Limitations

ATHIX is an ongoing prototype. Its results should be interpreted with care.

- Pose estimation accuracy depends on camera position, lighting, visibility and video quality.
- A 2D pose model does not provide complete 3D biomechanics measurements.
- The system currently focuses on body landmarks and joint angles; it does not provide validated bat or ball tracking.
- Movement-phase detection is experimental and may misclassify short or low-motion recordings.
- Similarity percentages are based on estimated joint-angle patterns and are not validated coaching scores.
- Differences in camera perspective, player orientation and recording conditions can affect comparisons.
- A single trainer reference cannot represent every valid batting style.

ATHIX is intended as an experimental analysis and visualisation tool, not a replacement for professional coaching or sports-science assessment.

---

## Roadmap

The project will evolve as the analysis pipeline becomes more reliable.

- [x] Initial cricket batting analysis prototype
- [x] YOLO-based human pose extraction
- [x] Joint-angle calculation
- [x] Angle smoothing
- [x] Basic movement-phase detection
- [x] Trainer reference saving and experimental comparison
- [x] Annotated pose-video visualisation
- [ ] Improve movement-phase detection
- [ ] Improve pose-estimation reliability
- [ ] Add more cricket batting techniques
- [ ] Improve reference alignment and comparison
- [ ] Develop more interpretable technique feedback
- [ ] Expand towards additional sports

---

## Contributing

ATHIX is currently under active development.

Suggestions, bug reports and ideas for improving the analysis pipeline are welcome.

If you would like to contribute:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Test your changes.
5. Submit a pull request describing your contribution.

---

## Project Status

**ATHIX is an experimental prototype under active development.**

The current priority is improving the reliability of pose extraction, movement analysis and reference-based comparison before expanding the platform.

---

## Author

**Manjunatha T V**

Creator and developer of ATHIX.

- GitHub: [@manjunatha-tv](https://github.com/manjunatha-tv)

---

<p align="center">
  <strong>ATHIX</strong><br>
  Making sports movement easier to understand through computer vision.
</p>
