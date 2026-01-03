import os
import sys
import time
import subprocess
import threading
import itertools

# Constants
ICON_PATH = os.path.join("assets", "icon", "z.png")
DIST_DIR = "dist"

class BuildManager:
    def __init__(self):
        self.start_time = time.time()
        self.steps = [
            ("Tests & Benchmarks", 10),    # Weight 10
            ("Building Main EXE", 60),     # Weight 60 (Slowest)
            ("Building Installer", 30)     # Weight 30
        ]
        self.total_weight = sum(w for _, w in self.steps)
        self.completed_weight = 0

    def print_header(self):
        print("\033[92m" + "="*50)
        print("      Zertam Vision - Smart Build System")
        print("="*50 + "\033[0m")

    def update_progress(self, current_step_name, current_step_weight, elapsed_step):
        # Rough ETA Calc
        # We assume 1 weight unit takes approx X seconds based on history?
        # Or simpler: total_time / completed_weight * remaining_weight
        
        now = time.time()
        total_elapsed = now - self.start_time
        
        # Avoid div zero
        effective_weight = self.completed_weight + (elapsed_step * 0.1) # Hacky progress within step
        if effective_weight <= 0: effective_weight = 0.1
        
        est_total = (total_elapsed / effective_weight) * self.total_weight
        remaining = est_total - total_elapsed
        if remaining < 0: remaining = 0
        
        mins, secs = divmod(int(remaining), 60)
        
        sys.stdout.write(f"\r\033[94m[Build] {current_step_name} ... ETA: {mins}m {secs}s\033[0m")
        sys.stdout.flush()

    def run_step(self, name, weight, command):
        stop_event = threading.Event()
        t = threading.Thread(target=self._spinner, args=(stop_event, name, weight))
        t.start()
        
        try:
            # Run
            # subprocess.run blocks, so if we get KeyboardInterrupt, it happens here
            status = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stop_event.set()
            t.join()
            
            if status.returncode != 0:
                print(f"\n\033[91m[FAIL] {name}\033[0m")
                print(status.stderr.decode())
                return False
                
            print(f"\r\033[92m[DONE] {name}                 \033[0m")
            self.completed_weight += weight
            return True
            
        except KeyboardInterrupt:
            stop_event.set()
            t.join()
            print(f"\n\033[93m[CANCEL] Build Cancelled by User.\033[0m")
            sys.exit(1)
        except Exception as e:
            stop_event.set()
            t.join()
            print(f"\nError: {e}")
            return False

    def _spinner(self, stop_event, name, weight):
        spinner = itertools.cycle(['-', '/', '|', '\\'])
        start_step = time.time()
        while not stop_event.is_set():
            self.update_progress(name, weight, time.time() - start_step)
            time.sleep(0.1)

def main():
    bm = BuildManager()
    bm.print_header()
    
    # 1. Tests
    if not bm.run_step("Tests & Benchmarks", 10, "pytest tests/"):
        return

    # 2. Main EXE (Directory Mode for Speed)
    print(f"\n[Build] Building ZertamVision (Dir Mode) (Icon: {ICON_PATH})")
    # --onedir creates dist/ZertamVision/ containing exe and internal folders
    cmd_main = f'python -m PyInstaller main.py --name ZertamVision --onedir --windowed --noconfirm --add-data "assets;assets" --add-data "utils;utils" --icon="{ICON_PATH}"'
    if not bm.run_step("Building Main App (Onedir)", 50, cmd_main):
        return

    # 3. Zip the Directory
    print(f"\n[Build] Zipping Application Bundle")
    # Create dist/ZertamVision.zip from dist/ZertamVision
    import shutil
    try:
        app_dir = os.path.join(DIST_DIR, "ZertamVision")
        zip_path = os.path.join(DIST_DIR, "ZertamVision") # shutil.make_archive adds .zip
        shutil.make_archive(zip_path, 'zip', app_dir)
        print(f"\r\033[92m[DONE] Zipping Bundle              \033[0m")
    except Exception as e:
        print(f"\nError zipping: {e}")
        return

    # 4. Installer (Bundle the ZIP)
    # Bundle dist/ZertamVision.zip
    zip_file = os.path.join(DIST_DIR, "ZertamVision.zip")
    if not os.path.exists(zip_file):
        print("Error: App Zip missing.")
        return
        
    cmd_installer = f'python -m PyInstaller Installer/installer.py --name ZertamInstaller --onefile --windowed --noconfirm --add-data "{zip_file};." --add-data "README.md;." --add-data "LICENSE;." --icon="{ICON_PATH}"'
    if not bm.run_step("Building Installer", 30, cmd_installer):
        return

    print(f"\n\033[92mBuild Complete in {int(time.time() - bm.start_time)}s!\033[0m")
    print(f"Output: {os.path.abspath(DIST_DIR)}")

if __name__ == "__main__":
    # Enable ANSI support on Windows 10/11
    os.system("") 
    main()
