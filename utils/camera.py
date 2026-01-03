import cv2
import threading
import time
from utils.logger import setup_logger

logger = setup_logger(name="Camera")

class ThreadedCamera:
    """
    A Threaded Camera wrapper to decouple frame reading from processing.
    This ensures that image processing (detection) does not block camera I/O.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            logger.error(f"Failed to open camera source: {src}")
            raise ValueError(f"Could not open video source {src}")

        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()
        
        # Performance metrics
        self.fps_start_time = time.time()
        self.fps_frames = 0
        self.fps = 0.0

    def start(self):
        """Starts the thread to read frames from the video stream."""
        logger.info("Starting camera thread")
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        """Loop that continuously polls for new frames."""
        while True:
            if self.stopped:
                return

            (grabbed, frame) = self.stream.read()
            
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame
                
            if not grabbed:
                logger.warning("Camera stream ended or failed to grab frame.")
                self.stop()
                return

            # Update FPS calculation
            self.fps_frames += 1
            if self.fps_frames % 30 == 0:
                self.fps = 30 / (time.time() - self.fps_start_time)
                self.fps_start_time = time.time()

    def read(self):
        """Return the most recent frame."""
        with self.lock:
            return self.frame

    def stop(self):
        """Stops the thread."""
        self.stopped = True
        self.stream.release()
        logger.info("Camera stopped")

    def get_fps(self):
        return self.fps
