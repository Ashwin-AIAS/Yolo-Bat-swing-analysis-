# Bat Swing Analysis ⚾

This is a complete mini-project that implements a bat swing analysis pipeline from a single-camera video. It detects the player and bat, tracks key body joints and the bat-tip, and computes advanced metrics for each swing.

## Features

* **Player & Bat Detection:** Uses a pre-trained YOLOv8 model.
* **Pose Estimation:** Uses MediaPipe to find shoulder, elbow, wrist, and hip keypoints.
* **Bat-Tip Tracking:** Estimates the bat-tip position based on bat detection and wrist position.
* **Swing Detection:** Automatically identifies swing events based on bat-tip speed.
* **Core Metrics (per swing):**
    * **Peak Linear Speed (m/s):** Max speed of the bat-tip.
    * **Peak Angular Velocity (rad/s):** Max angular velocity of the `shoulder -> bat-tip` vector.
    * **Swing Duration (s):** Total time of the swing.
    * **Swing Angle (deg):** Total angular displacement of the `shoulder -> bat-tip` vector.
    * **Smoothness Score:** A score from 0 (jerky) to 1 (smooth) based on the spectral entropy of the bat-tip's acceleration.
* **Outputs:**
    * `metrics.csv`: A CSV file with all metrics for every detected swing.
    * Visuals (in output directory):
        * `swing_{id}_overlay.gif`: An animated GIF of the swing with overlays.
        * `swing_{id}_plots.png`: A plot of linear speed and angular velocity over time.

---

## 🚀 Quick Start

### 1. Setup

**Prerequisites:**
* Python 3.10+
* FFmpeg (for creating GIFs). On macOS: `brew install ffmpeg`. On Ubuntu: `sudo apt install ffmpeg`.

**Installation:**

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd bat-swing-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash  
    python3 -m venv venv
    source venv/bin/activate
    ```  

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt  
    ```

4.  **Download a sample video:**
    * Download a sample baseball swing video. A good one is available from [Pexels (Video by Tima Miroshnichenko)](https://www.pexels.com/video/man-swinging-bat-at-ball-in-cage-5343153/).
    * Download it and save it as `data/sample_videos/swing1.mp4`.

### 2. Run via Command Line (CLI)

The `run_analysis.py` script is the main entrypoint.

**Calibration (`--scale`):**
You *must* provide a pixel-to-meter scale. To find this:
1.  Open your video and find a frame with an object of known length (e.g., the bat, $\approx$ 1.0m).
2.  Use an image editor to measure the pixel length of that object.
3.  Calculate the scale: `scale = known_meters / measured_pixels`.
    * *Example:* If a 1.0m bat is 250 pixels long, the scale is `1.0 / 250 = 0.004`.

**Run:**

```bash
# Example command using an estimated scale
python run_analysis.py \
    --video data/sample_videos/swing1.mp4 \
    --scale 0.004 \
    --output_dir results/swing1_analysis
```

This will create the `results/swing1_analysis` directory and fill it with `metrics.csv`, `*.gif`, and `*.png` files.

### 3. Run via Streamlit Demo

A simple web-based demo is also provided.

```bash
streamlit run streamlit_app.py
```

Open the provided URL in your browser. You can upload a video, set the scale, and run the analysis.

---

## 🧪 Unit Tests

To ensure the metric calculations are correct, you can run the unit tests:

```bash
pytest
```

---

## 🏋️ Training a Custom YOLOv8 Model

The default `yolov8n.pt` model is decent at detecting `person` but may struggle with `bat`. For best results, you should fine-tune a model on your own data.

**Classes to Annotate:**
* `person`: Bounding box for the player.
* `bat`: Bounding box for the bat.
* `bat_tip` (Optional, but *highly recommended*): A bounding box around just the tip of the bat. If our code detects this class, it will use it directly, skipping the estimation heuristic.

**Workflow:**
1.  **Collect Data:** Record videos or find sample videos of bat swings.
2.  **Extract Frames:** Use a script (or FFmpeg) to export frames from your videos.
3.  **Annotate:** Use a tool like [Roboflow](https://roboflow.com/) or [CVAT](https://github.com/cvat-ai/cvat) to draw bounding boxes for the classes above.
4.  **Export:** Export your dataset in **YOLOv8 (YOLOv5 PyTorch)** format. This will give you a `data.yaml` file and `train/`, `valid/` directories with images and labels.
5.  **Train:**
    ```bash
    # Install ultralytics if not already installed (it is in requirements.txt)
    # pip install ultralytics

    # Run training
    yolo train data=/path/to/your/data.yaml model=yolov8n.pt epochs=100 imgsz=640
    ```
6.  **Use Model:** The trained model will be saved in `runs/detect/train/weights/best.pt`. Use this path for the `--yolo-model` argument in `run_analysis.py`.

---

## 📁 Project Structure

```
bat-swing-analysis/
├── README.md
├── requirements.txt         # Project dependencies
├── run_analysis.py          # Main CLI entrypoint
├── streamlit_app.py         # Streamlit web demo
├── detectors/
│   └── yolov8_detector.py   # YOLOv8 detection wrapper
├── pose/
│   └── mediapipe_pose.py    # MediaPipe pose estimation wrapper
├── tracker/
│   └── sort_wrapper.py      # SORT tracker wrapper (Note: not used in simplified single-player pipeline)
├── metrics/
│   └── swing_metrics.py     # Core logic for swing detection and metric calculation
├── viz/
│   ├── overlay.py           # Functions to draw overlays and create GIFs
│   └── plots.py             # Functions to plot metrics
├── utils/
│   └── geometry.py          # Helper functions for interpolation, smoothing, angles
├── tests/
│   └── test_metrics.py      # Unit tests for core metric logic
└── data/
    └── sample_videos/
        └── .gitkeep         # Placeholder for sample videos
```
