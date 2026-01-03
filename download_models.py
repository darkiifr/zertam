import urllib.request
import os
import sys

MODELS_DIR = "models"
FILES = {
    "MobileNetSSD_deploy.prototxt": "https://raw.githubusercontent.com/robmarkcole/object-detection-app/master/model/MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://raw.githubusercontent.com/robmarkcole/object-detection-app/master/model/MobileNetSSD_deploy.caffemodel"
}

def download_models():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"Created directory: {MODELS_DIR}")

    for filename, url in FILES.items():
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"File {filename} already exists. Skipping.")
            continue

        print(f"Downloading {filename} from {url}...")
        try:
            # Set a generic user agent to avoid git blocking
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            
            urllib.request.urlretrieve(url, path)
            size = os.path.getsize(path)
            print(f"Downloaded {filename} ({size} bytes)")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    download_models()
