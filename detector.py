import cv2
import numpy as np
import os
from utils.logger import setup_logger

logger = setup_logger(name="Detector")

class ObjectDetector:
    def __init__(self, model_path, config_path, confidence_threshold=0.5, preprocessing=None):
        self.confidence_threshold = confidence_threshold
        self.preprocessing = preprocessing or {
            "size": (300, 300),
            "scale": 0.007843,
            "mean": (127.5, 127.5, 127.5),
            "swapRB": False
        }
        
        # Default classes for MobileNet SSD (COCO based 20 classes)
        # TODO: Ideally this should also be dynamic per model, but keeping as is for now for MobileNet
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train", "tvmonitor"]
        
        # Assign random colors to each class
        np.random.seed(42)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Loading config from: {config_path}")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            logger.error("Model files not found!")
            raise FileNotFoundError("Model or Config file not found. Please check paths.")

        try:
            self.net = cv2.dnn.readNet(config_path, model_path)
            # Try to use hardware acceleration if available (e.g. OpenCL)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e

    def detect(self, frame):
        """
        Performs object detection on the frame.
        Returns the specific bounding boxes and labels.
        """
        (h, w) = frame.shape[:2]
        
        # Use dynamic preprocessing params
        size = self.preprocessing.get("size", (300, 300))
        scale = self.preprocessing.get("scale", 0.007843)
        mean = self.preprocessing.get("mean", (127.5, 127.5, 127.5))
        swapRB = self.preprocessing.get("swapRB", False)

        blob = cv2.dnn.blobFromImage(frame, scale, size, mean, swapRB)

        self.net.setInput(blob)
        detections = self.net.forward()
        
        # YOLO models return a different shape, we might need to handle that if fully supporting YOLO
        # MobileNet/Caffe returns [1, 1, N, 7]
        # YOLOv3 returns a list of layer outputs.
        # For now, let's assume we are sticking to models that work with `forward()` returning standard detection output
        # OR we need to add robust handling. 
        # Given constraints, we will stick to the current logic which works for MobileNet/FaceDetector
        # YOLOv3 usually needs `forward(output_names)` and post-processing (NMS).
        
        results = []

        if len(detections.shape) == 4: # Standard SSD/Caffe output
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > self.confidence_threshold:
                    idx = int(detections[0, 0, i, 1])
                    
                    # Check bounds
                    if idx < 0: continue
                    # For Face Detector, class 1 might be face, but index might vary.
                    # MobileNet SSD has background at 0.
                    
                    label_text = "Object"
                    if idx < len(self.CLASSES):
                         label_text = self.CLASSES[idx]
                    else:
                         label_text = f"Class {idx}"

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    display_label = f"{label_text}: {confidence * 100:.2f}%"
                    
                    # Color
                    color = self.COLORS[idx] if idx < len(self.COLORS) else (0, 255, 0)
                    
                    results.append({
                        "label": label_text,
                        "confidence": float(confidence),
                        "box": (startX, startY, endX, endY),
                        "color": color,
                        "display_label": display_label
                    })
        else:
             # Placeholder for other model types if the shap is different
             pass

        return results
