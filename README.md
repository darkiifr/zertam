# Zertam Vision

A powerful, modular Computer Vision application built with Python, Dear PyGui, and OpenCV.

## Features
- **Real-time Object Detection**: Support for MobileNet SSD, YOLO, and more.
- **Model Marketplace**: Download and switch between different AI models instantly.
- **Train Studio**: Create your own datasets (Initial Camera Capture & Video Extraction).
- **Object Tracking**: Track unique objects across frames.
- **Hardware Monitoring**: Real-time CPU & GPU usage statistics.
- **Gallery**: View and manage your snapshots.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/zertam.git
   cd zertam
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Requires `opencv-python`, `dearpygui`, `numpy`, `psutil`, `scipy`.*

3. **Run the application**:
   ```bash
   python main.py
   ```

## Usage
- **Camera View**: Main interface for detection. Use the sidebar to switch models or change settings.
- **Marketplace**: Download new models.
- **Train Studio**: Capture images for training custom models.
- **Gallery**: Review your snapshots.

## License
MIT License. See [LICENSE](LICENSE) for details.
