import pytest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import ObjectDetector

# Fixture to provide a detector instance
@pytest.fixture(scope="module")
def detector_instance():
    # Use MobileNet as it's small/fast
    m_path = "models/MobileNetSSD_deploy.caffemodel"
    c_path = "models/MobileNetSSD_deploy.prototxt"
    
    # Skip if files don't exist (e.g. CI without download)
    if not os.path.exists(m_path):
        pytest.skip("Model files not found")

    preprocessing = {
        "size": (300, 300),
        "scale": 0.007843,
        "mean": (127.5, 127.5, 127.5),
        "swapRB": False
    }
    return ObjectDetector(m_path, c_path, preprocessing=preprocessing)

@pytest.fixture
def dummy_frame():
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

def test_inference_benchmark(benchmark, detector_instance, dummy_frame):
    """Benchmarks the detector's inference speed."""
    
    # benchmark() runs the function many times and records stats
    benchmark(detector_instance.detect, dummy_frame)

def test_detector_initialization(detector_instance):
    assert detector_instance.net is not None
