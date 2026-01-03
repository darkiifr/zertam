import cv2
import time
import numpy as np
import os
import argparse
from detector import ObjectDetector

def run_benchmark(model_path, config_path, iterations=100):
    print(f"Benchmarking Model: {model_path}")
    print(f"Iterations: {iterations}")

    try:
        # 1. Load Model
        start_load = time.time()
        # Default preprocessing for benchmark (MobileNet like)
        preprocessing = {
            "size": (300, 300),
            "scale": 0.007843,
            "mean": (127.5, 127.5, 127.5),
            "swapRB": False
        }
        detector = ObjectDetector(model_path, config_path, preprocessing=preprocessing)
        load_time = time.time() - start_load
        print(f"Model Load Time: {load_time:.4f}s")
        
        # 2. Prepare Dummy Input
        frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        # 3. Warmup
        print("Warming up...")
        for _ in range(5):
            detector.detect(frame)
            
        # 4. Run Loop
        print("Running inference...")
        latencies = []
        for i in range(iterations):
            t0 = time.time()
            detector.detect(frame)
            t1 = time.time()
            latencies.append(t1 - t0)
            
        # 5. Report
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        min_lat = min(latencies)
        fps = 1.0 / avg_lat
        
        print("\n--- Benchmark Results ---")
        print(f"Average FPS: {fps:.2f}")
        print(f"Avg Latency: {avg_lat*1000:.2f}ms")
        print(f"Max Latency: {max_lat*1000:.2f}ms")
        print(f"Min Latency: {min_lat*1000:.2f}ms")
        print("-------------------------")
        
    except Exception as e:
        print(f"Benchmark Failed: {e}")

if __name__ == "__main__":
    # Default to MobileNet if exists
    m_path = "models/MobileNetSSD_deploy.caffemodel"
    c_path = "models/MobileNetSSD_deploy.prototxt"
    
    if os.path.exists(m_path) and os.path.exists(c_path):
        run_benchmark(m_path, c_path)
    else:
        print(f"Models not found at {m_path}. Please download them first.")
