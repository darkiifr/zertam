import cv2
import argparse
import time
import os
import sys
from detector import ObjectDetector
from utils.camera import ThreadedCamera
from utils.logger import setup_logger

# Parse Arguments
parser = argparse.ArgumentParser(description="Zertam Vision: Lightweight Object Detection")
parser.add_argument("--model", type=str, default="models/MobileNetSSD_deploy.caffemodel",
                    help="Path to the Caffe model file")
parser.add_argument("--config", type=str, default="models/MobileNetSSD_deploy.prototxt",
                    help="Path to the Caffe config (prototxt) file")
parser.add_argument("--camera", type=int, default=0, help="Camera Index (default 0)")
parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--skip", type=int, default=1, help="Process every Nth frame (default 1 = every frame)")
args = parser.parse_args()

logger = setup_logger()

def main():
    logger.info("Initializing Zertam Vision...")

    # Validate model paths
    if not os.path.exists(args.model) or not os.path.exists(args.config):
        logger.error(f"Model files not found at {args.model} or {args.config}")
        print("Error: Model files not found. Please checks paths or download them.")
        return

    # Initialize Detector
    try:
        detector = ObjectDetector(args.model, args.config, args.confidence)
    except Exception as e:
        logger.critical(f"Failed to initialize detector: {e}")
        return

    # GUI Mode Logic
    logger.info("Launching GUI Mode...")
    # Import here to avoid imports if not needed (though we need them now)
    try:
        from gui_app import ZertamApp
        app = ZertamApp(args.model, args.config)
        app.run()
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.critical(f"GUI Crash: {e}")
        # print(f"Failed to launch GUI: {e}")
    
    return

    # OLD CLI LOGIC BELOW (Archived/Removed for now as user requested switch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Zertam] Application stopped by user.")
    except Exception as e:
        print(f"\n[Zertam] Critical Error: {e}")
