  # ATHIX — AI-Assisted Sports Technique Analysis

<p align="center">
  <strong>Understand movement. Analyse technique. Improve performance.</strong>
</p>

<p align="center">
  An experimental computer-vision platform for analysing cricket batting technique through video-based human pose estimation and biomechanical measurements.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Prototype-orange" alt="Project status">
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


## Key Features

### 1. Video-Based Analysis

### 2. YOLO Pose Estimation

### 3. Joint-Angle Calculation

Calculates selected biomechanical angles, including:

- Left and right elbow angles
- Left and right knee angles
- Left and right shoulder angles
- Left and right hip angles

### 4. Movement Data Processing

### 5. Annotated Video Visualisation

### 6. Experimental Similarity Measurement

ATHIX calculates an experimental similarity percentage between the player's recording and the saved trainer reference.

The comparison currently uses eight joint-angle patterns and 100 normalised time samples.

**Important:** This percentage represents similarity to the selected reference recording. It is not a scientifically validated technique score or a professional coaching grade.

### 8. 3D Cricket Visualisation

Includes a 3D cricket batter model as part of the technique visualisation experience.

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
## Author

**Manjunatha T V**

Creator and developer of ATHIX.

- GitHub: [@manjunatha-tv](https://github.com/manjunatha-tv)

---
