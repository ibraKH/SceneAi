# SceneAI

## Project Overview

**SceneAI** is a local, real-time scene understanding system built for Apple Silicon.

It opens the user’s camera, detects and tracks visible objects, understands changes in the scene, and generates short Arabic descriptions through a clean right-to-left dashboard.

All camera frames and AI inference stay on the device.

---

## Tech Stack

### Frontend

- **Interface:** HTML, CSS, and JavaScript
- **Design:** Responsive Arabic RTL dashboard
- **Live Video:** Browser Camera API
- **Visualization:** Canvas bounding boxes and object labels
- **Communication:** WebSocket streaming

### Backend

- **Language:** Python
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **Server:** [Uvicorn](https://www.uvicorn.org/)
- **Transport:** Real-time WebSocket connection

### AI & Computer Vision

- **Object Detection:** [RF-DETR Nano](https://github.com/roboflow/rf-detr)
- **Scene Tracking:** Lightweight IOU object tracker
- **Arabic Scene Description:** Qwen2.5-VL-3B Instruct (4-bit)
- **AI Runtime:** [PyTorch](https://pytorch.org/) with Apple Metal and [MLX](https://github.com/ml-explore/mlx)
- **Image Processing:** Pillow and NumPy

---

## Key Features

### 1. Real-Time Object Detection

- Detects objects directly from the live camera.
- Displays bounding boxes, Arabic labels, and confidence scores.
- Uses Apple Metal acceleration with automatic CPU fallback.

### 2. Scene Awareness

- Assigns a stable ID to each visible object.
- Tracks when objects appear, disappear, or change in number.
- Maintains scene context instead of treating every frame independently.

### 3. Arabic Scene Descriptions

- Generates short, contextual descriptions in Arabic.
- Updates when something meaningful changes in the scene.
- Avoids repeatedly describing an unchanged view.

### 4. Local-First Privacy

- Camera frames never leave the device.
- Detection, tracking, and language generation run locally.
- No cloud inference API is required.

---

## Getting Started

### Requirements

- Apple Silicon Mac (M1 or newer)
- macOS 13 or newer
- Python 3.11–3.13
- At least 8 GB of memory
- A webcam and a modern browser
- Internet access for the initial model download

### Setup

```bash
git clone https://github.com/ibraKH/SceneAi.git
cd SceneAi
./setup.sh
```

The first setup downloads the required Python packages and AI model weights.

### Run

```bash
./run.sh
```

Open [http://localhost:8000](http://localhost:8000) and allow camera access when prompted.

For Macs with limited memory, use the smaller vision-language model:

```bash
VLM_MODEL_ID=mlx-community/Qwen2-VL-2B-Instruct-4bit ./run.sh
```

---

## Tests

The test suite runs without loading the AI models or using the network.

```bash
.venv/bin/python -m pytest
```
