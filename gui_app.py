import dearpygui.dearpygui as dpg
import cv2
import array
import numpy as np
import threading
import time
from datetime import datetime
import os
import logging
from detector import ObjectDetector, ImageClassifier
from utils.camera import ThreadedCamera
from utils.logger import setup_logger, add_gui_handler
from utils.hardware import get_cpu_name, get_gpu_name, get_system_stats
from utils.tracker import CentroidTracker
from utils.updater import check_for_updates
from utils.marketplace import AVAILABLE_MODELS, get_model_status, install_model
import psutil

import queue

# Constants
WIDTH = 1280
HEIGHT = 720
TEXTURE_TAG = "cv_frame"
GALLERY_TEXTURE_TAG = "gallery_frame"
FONT_REGULAR = "assets/fonts/Figtree-Regular.ttf"

class GuiLogHandler(logging.Handler):
    """Custom logging handler to send logs to a queue."""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)

class ZertamApp:
    def __init__(self, model_path, config_path):
        self.logger = setup_logger()
        self.camera = None
        self.detector = None
        self.running = True
        
        self.frame_data = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
        self.gallery_image_data = np.zeros((480, 640, 4), dtype=np.float32) # Default placeholder
        
        # Stats
        self.fps_history = [0.0] * 100
        
        # Settings
        self.confidence = 0.5
        self.skip_frames = 1
        self.current_model_id = "mobilenet_ssd" # Default ID
        
        # Log Queue
        self.log_queue = queue.Queue()
        
        
        # Hardware Info
        self.cpu_name = get_cpu_name()
        self.gpu_name = get_gpu_name()
        
        # Advanced Features
        self.tracker = CentroidTracker()
        self.tracking_enabled = False
        self.train_class_name = "NewObject"
        self.classifier = None
        self.helper_enabled = False
        
        # Initialize Core
        try:
            # Find default model meta
            default_model = next((m for m in AVAILABLE_MODELS if m["id"] == self.current_model_id), AVAILABLE_MODELS[0])
            self.reload_model(model_path, config_path, default_model["name"], default_model.get("preprocessing"))
            self.camera = ThreadedCamera(0).start()
            time.sleep(1.0) # Warmup
        except Exception as e:
            self.logger.critical(f"Init failed: {e}")
            
        self.setup_gui()
        
        # Add GUI Logger
        gui_handler = GuiLogHandler(self.log_queue)
        add_gui_handler(self.logger, gui_handler)
        
        # Start hardware stats timer
        self.last_stats_update = 0

    def reload_model(self, model_path, config_path, name, preprocessing=None):
        """Hot-reloads the object detector."""
        self.logger.info(f"Switching model to: {name}...")
        try:
            # Pause detection if running
            old_detector = self.detector
            self.detector = None # Stop using it in main loop
            
            # Load new
            new_detector = ObjectDetector(model_path, config_path, self.confidence, preprocessing)
            self.detector = new_detector
            self.logger.info(f"Model {name} loaded successfully!")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model {name}: {e}")
            self.detector = old_detector # Revert
            return False

    def setup_gui(self):
        dpg.create_context()
        dpg.create_viewport(title="Zertam Vision - Pro", width=1400, height=900)
        dpg.setup_dearpygui()
        
        # Load Fonts
        with dpg.font_registry():
            try:
                if os.path.exists(FONT_REGULAR):
                    self.font_regular = dpg.add_font(FONT_REGULAR, 18)
                    dpg.bind_font(self.font_regular)
                else:
                    self.logger.warning("Custom font not found, using default.")
            except Exception as e:
                self.logger.error(f"Font loading error: {e}")

        # Setup Theme
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 8)
                dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 5)
                dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 5)
                
                # Colors
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (25, 25, 25))
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (35, 35, 35))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (60, 60, 60))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 40, 40))
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 255, 128))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (0, 255, 128))
                
        dpg.bind_theme(global_theme)

        # Texture Registry
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(width=WIDTH, height=HEIGHT, default_value=self.frame_data, format=dpg.mvFormat_Float_rgba, tag=TEXTURE_TAG)
            dpg.add_raw_texture(width=640, height=480, default_value=self.gallery_image_data, format=dpg.mvFormat_Float_rgba, tag=GALLERY_TEXTURE_TAG)

        # Snapshot Status
        self.snapshot_message = ""
        self.snapshot_timer = 0
        
        # Check Updates
        check_for_updates(self.on_update_found)

        # Main Window
        with dpg.window(label="Main Window", tag="primary_window"):
            
            with dpg.tab_bar(tag="main_tab_bar"):
                
                # TAB 1: Camera View
                with dpg.tab(label="Camera View"):
                    with dpg.group(horizontal=True):
                        # Viewport content
                        dpg.add_image(TEXTURE_TAG, width=900, height=675)
                        
                        # Right Side Panel
                        with dpg.child_window(width=450, height=675):
                            dpg.add_text("System Hardware", color=(0, 255, 128))
                            dpg.add_text(f"CPU: {self.cpu_name}")
                            dpg.add_text("Usage: ...", tag="lbl_cpu_usage")
                            dpg.add_text(f"GPU: {self.gpu_name}")
                            dpg.add_text("RAM: ...", tag="lbl_ram_usage")
                            
                            dpg.add_separator()
                            dpg.add_text("Model Selection")
                            # Combo box for model selection
                            model_names = [m["name"] for m in AVAILABLE_MODELS]
                            dpg.add_combo(items=model_names, default_value=model_names[0], callback=self.on_model_combo_change, tag="combo_models")
                            dpg.add_text("", tag="lbl_model_status", color=(255, 255, 0))
                            
                            dpg.add_separator()
                            dpg.add_button(label="Snapshot", width=-1, height=40, callback=self.on_snapshot_click) # Text fixed
                            dpg.add_text("", tag="lbl_snapshot_status", color=(0, 255, 255))
                            
                            dpg.add_separator()
                            dpg.add_text("Settings")
                            dpg.add_slider_float(label="Confidence", default_value=0.5, max_value=1.0, callback=self.update_conf)
                            dpg.add_slider_int(label="Skip Frames", default_value=1, min_value=1, max_value=10, callback=self.update_skip)
                            dpg.add_checkbox(label="Enable Tracking", callback=self.toggle_tracking)
                            dpg.add_checkbox(label="Fullscreen", callback=self.toggle_fullscreen)
                            
                            dpg.add_separator()
                            dpg.add_text("Performance")
                            with dpg.plot(label="FPS", height=120, width=-1):
                                dpg.add_plot_legend()
                                dpg.add_plot_axis(dpg.mvXAxis, label="", tag="x_axis")
                                dpg.add_plot_axis(dpg.mvYAxis, label="FPS", tag="y_axis")
                                dpg.add_line_series(list(range(100)), self.fps_history, label="FPS", tag="fps_series", parent="y_axis")
                            
                            dpg.add_separator()
                            with dpg.group(horizontal=True):
                                dpg.add_text("Logs")
                                dpg.add_button(label="Clear", callback=self.on_clear_logs_click)
                            with dpg.child_window(tag="log_list", height=150, width=-1, border=True):
                                dpg.add_text("--- System Started ---")

                # TAB 2: Marketplace
                with dpg.tab(label="Model Marketplace"):
                    dpg.add_text("Download Models from Cloud", color=(100, 200, 255))
                    dpg.add_text("Ready", tag="lbl_download_status")
                    
                    with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
                        dpg.add_table_column(label="Name", width_stretch=True, init_width_or_weight=0.2)
                        dpg.add_table_column(label="Description", width_stretch=True, init_width_or_weight=0.4)
                        dpg.add_table_column(label="Status", width_stretch=True, init_width_or_weight=0.15)
                        dpg.add_table_column(label="Action", width_stretch=True, init_width_or_weight=0.25)

                        for model in AVAILABLE_MODELS:
                            with dpg.table_row():
                                dpg.add_text(model["name"])
                                dpg.add_text(model["description"], wrap=300)
                                status = get_model_status(model["id"])
                                col = (0, 255, 128) if status == "Installed" else (255, 100, 100)
                                
                                # Check if tag exists to avoid duplicate alias error on reload/redraw
                                tag_id = f"status_{model['id']}"
                                if dpg.does_item_exist(tag_id):
                                    dpg.delete_item(tag_id)
                                    
                                dpg.add_text(status, color=col, tag=tag_id)
                                dpg.add_button(label="Download", callback=self.on_download_click, user_data=model["id"])

                # TAB 3: Train Studio
                with dpg.tab(label="Train Studio"):
                    dpg.add_text("Create Datasets for Training", color=(0, 255, 128))
                    dpg.add_separator()
                    
                    with dpg.tab_bar():
                        # Sub-tab: Camera Capture
                        with dpg.tab(label="Camera Capture"):
                             with dpg.group(horizontal=True):
                                 # Preview
                                 dpg.add_image(TEXTURE_TAG, width=640, height=480)
                                 
                                 # Controls
                                 with dpg.group():
                                     dpg.add_text("Class Name:")
                                     dpg.add_input_text(default_value="NewObject", tag="input_class_name", callback=self.update_train_name)
                                     dpg.add_spacer(height=10)
                                     
                                     dpg.add_text("Target Architecture:")
                                     dpg.add_combo(items=["MobileNetSSD", "YOLOv3", "ResNet50", "Custom"], default_value="MobileNetSSD", width=200)
                                     
                                     dpg.add_separator()
                                     dpg.add_checkbox(label="Use Helper Classifier", callback=self.toggle_helper)
                                     class_models = [m["name"] for m in AVAILABLE_MODELS if "Classification" in m["description"]]
                                     dpg.add_combo(items=class_models, callback=self.on_classifier_combo_change, width=200, tag="combo_classifier")
                                     dpg.add_button(label="Download Model", tag="btn_download_helper", show=False, callback=self.on_download_helper_click)
                                     dpg.add_text("Prediction: None", tag="lbl_helper_pred", color=(255, 200, 0))
                                     
                                     dpg.add_spacer(height=10)
                                     dpg.add_button(label="Capture Image", width=200, height=50, callback=self.on_train_capture_click)
                                     dpg.add_text("", tag="lbl_train_status", color=(0, 255, 255))
                                     
                        # Sub-tab: Video Extraction
                        with dpg.tab(label="Video Extraction"):
                            dpg.add_text("Extract frames from video files.")
                            dpg.add_input_text(label="Video Path", tag="input_video_path")
                            dpg.add_button(label="Process Video", callback=self.on_video_process_click)
                            dpg.add_text("", tag="lbl_video_status")

                # TAB 4: Gallery
                with dpg.tab(label="Gallery"):
                    with dpg.group(horizontal=True):
                        with dpg.child_window(width=200):
                            dpg.add_text("Captures")
                            dpg.add_button(label="Refresh", callback=self.refresh_gallery)
                            dpg.add_listbox([], tag="gallery_list", num_items=20, callback=self.on_gallery_select)
                        
                        with dpg.child_window(width=-1):
                             dpg.add_text("Preview")
                             dpg.add_image(GALLERY_TEXTURE_TAG, width=640, height=480)

        dpg.set_primary_window("primary_window", True)
        dpg.show_viewport()
        
        # Initial refresh
        self.refresh_gallery()

    def update_hardware_stats(self):
        now = time.time()
        if now - self.last_stats_update > 2.0: # Update every 2s
            stats = get_system_stats()
            dpg.set_value("lbl_cpu_usage", f"CPU Usage: {stats['cpu']}%")
            dpg.set_value("lbl_ram_usage", f"RAM Usage: {stats['ram']}%")
            self.last_stats_update = now

    def on_model_combo_change(self, sender, app_data):
        # Find model by name
        model = next((m for m in AVAILABLE_MODELS if m["name"] == app_data), None)
        if model:
            self.logger.info(f"Selected model: {model['name']}")
            
            # Check if installed
            if get_model_status(model["id"]) != "Installed":
                dpg.set_value("lbl_model_status", "Model not installed! Please download in Marketplace.")
                return
            
            dpg.set_value("lbl_model_status", "Loading...")
            
            # Check RAM if needed
            if model.get("requires_high_ram", False):
                mem = psutil.virtual_memory()
                total_gb = mem.total / (1024**3)
                if total_gb < 8:
                    dpg.set_value("lbl_model_status", f"WARNING: Low RAM ({total_gb:.1f}GB). Model requires >8GB.")
                    # We allow proceeding but with warning logged
                    self.logger.warning(f"Loading High RAM model on {total_gb:.1f}GB system.")
            
            path_model = os.path.join("models", model["filename_model"])
            path_config = os.path.join("models", model["filename_config"])
            
            success = self.reload_model(path_model, path_config, model["name"], model.get("preprocessing"))
            if success:
                dpg.set_value("lbl_model_status", "")
                self.current_model_id = model["id"]
            else:
                dpg.set_value("lbl_model_status", "Failed to load.")

    def update_conf(self, sender, data):
        if self.detector: self.detector.confidence_threshold = data

    def update_skip(self, sender, data):
        self.skip_frames = data

    def on_download_click(self, sender, app_data, user_data):
        dpg.set_value("lbl_download_status", f"Downloading {user_data}...")
        self.logger.info(f"Starting download for {user_data}...")
        
        # Start download
        install_model(user_data, lambda success: self.on_download_complete(success, user_data))
        
    def on_download_complete(self, success, model_id):
        status_text = "Ready" if success else "Download Failed"
        # We need to queue this update usually because it's from another thread
        # But DPG handle basic item updates thread-safely often.
        try:
            dpg.set_value("lbl_download_status", status_text)
            if success == True or success == model_id: # install_model passes model_id in old version, check compliance
                 dpg.set_value(f"status_{model_id}", "Installed")
                 dpg.configure_item(f"status_{model_id}", color=(0, 255, 128))
        except:
            pass

    def on_snapshot_click(self, sender, app_data, user_data):
        if not os.path.exists("captures"):
            os.makedirs("captures")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"captures/snapshot_{timestamp}.png"
        try:
             if hasattr(self, 'last_frame') and self.last_frame is not None:
                cv2.imwrite(filename, self.last_frame)
                self.logger.info(f"Snapshot saved: {filename}")
                dpg.set_value("lbl_snapshot_status", "Snapshot Saved!")
                self.snapshot_timer = time.time()
                self.refresh_gallery() # Update gallery list
             else:
                self.logger.warning("No frame to capture")
        except Exception as e:
            self.logger.error(f"Snapshot failed: {e}")

    def on_clear_logs_click(self, sender, app_data, user_data):
        dpg.delete_item("log_list", children_only=True)
        dpg.add_text("--- Logs Cleared ---", parent="log_list")

    def toggle_fullscreen(self, sender, data):
        dpg.toggle_viewport_fullscreen()

    def toggle_tracking(self, sender, data):
        self.tracking_enabled = data

    def update_train_name(self, sender, data):
        self.train_class_name = data

    def toggle_helper(self, sender, data):
        self.helper_enabled = data
        if data and not self.classifier:
            # Try to load selected
            model_name = dpg.get_value("combo_classifier")
            if model_name:
                self.on_classifier_combo_change(None, model_name)

    def on_classifier_combo_change(self, sender, data):
        model = next((m for m in AVAILABLE_MODELS if m["name"] == data), None)
        if not model: return
        
        status = get_model_status(model["id"])
        if status != "Installed":
            dpg.set_value("lbl_helper_pred", "Model not installed!")
            dpg.configure_item("btn_download_helper", show=True, user_data=model["id"])
            return
        
        dpg.configure_item("btn_download_helper", show=False)
        self.load_classifier(model)

    def on_download_helper_click(self, sender, app_data, user_data):
        dpg.set_value("lbl_helper_pred", "Downloading...")
        dpg.configure_item("btn_download_helper", show=False)
        install_model(user_data, lambda mid: self.on_helper_download_complete(mid))

    def on_helper_download_complete(self, model_id):
        # Find model name to reload combo logic
        model = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
        if model:
            dpg.set_value("lbl_helper_pred", "Installed! Loading...")
            self.load_classifier(model)

    def load_classifier(self, model):
        try:
            path_m = os.path.join("models", model["filename_model"])
            path_c = os.path.join("models", model["filename_config"])
            self.classifier = ImageClassifier(path_m, path_c, model.get("preprocessing"))
            self.logger.info(f"Classifier {model['name']} loaded.")
            dpg.set_value("lbl_helper_pred", f"Loaded {model['name']}")
        except Exception as e:
            self.logger.error(f"Failed to load classifier: {e}")
            dpg.set_value("lbl_helper_pred", "Load Failed")

    def on_train_capture_click(self, sender, app_data):
        self._save_dataset_image(self.train_class_name)

    def _save_dataset_image(self, class_name, frame=None):
        if frame is None:
             if hasattr(self, 'last_frame') and self.last_frame is not None:
                 frame = self.last_frame
             else:
                 return

        base_dir = os.path.join("dataset", class_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(base_dir, f"{timestamp}.jpg")
        
        try:
            cv2.imwrite(filename, frame)
            dpg.set_value("lbl_train_status", f"Saved to dataset/{class_name}")
            self.logger.info(f"Train image saved: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save train image: {e}")

    def on_video_process_click(self, sender, data):
        video_path = dpg.get_value("input_video_path")
        if not os.path.exists(video_path):
            dpg.set_value("lbl_video_status", "File not found!")
            return
            
        dpg.set_value("lbl_video_status", "Processing...")
        
        # Run in thread to not block UI
        def _process():
            cap = cv2.VideoCapture(video_path)
            count = 0
            saved = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Save every 30th frame (approx 1 sec)
                if count % 30 == 0:
                    self._save_dataset_image(self.train_class_name, frame)
                    saved += 1
                count += 1
            cap.release()
            dpg.set_value("lbl_video_status", f"Done! Extracted {saved} frames.")
            
        threading.Thread(target=_process, daemon=True).start()

    # Gallery
    def refresh_gallery(self):
        if not os.path.exists("captures"): return
        files = [f for f in os.listdir("captures") if f.endswith(".png")]
        files.sort(reverse=True) # Newest first
        dpg.configure_item("gallery_list", items=files)

    def on_gallery_select(self, sender, app_data):
        path = os.path.join("captures", app_data)
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = cv2.resize(img, (640, 480))
            data = img.astype(np.float32) / 255.0
            dpg.set_value(GALLERY_TEXTURE_TAG, data.flatten())

    def run(self):
        frame_count = 0
        last_results = []
        
        while dpg.is_dearpygui_running():
            self.update_hardware_stats()
            
            if self.camera:
                frame = self.camera.read()
            else:
                frame = None

            ui_only = False
            if frame is None:
                ui_only = True

            if not ui_only:
                frame_count += 1
                self.last_frame = frame.copy()
                
                # Detect
                if self.detector and frame_count % self.skip_frames == 0:
                    try:
                        last_results = self.detector.detect(frame)
                    except Exception as e:
                        # Log once to avoid spam
                        if frame_count % 100 == 0:
                            self.logger.error(f"Detection Error: {e}")
                        last_results = []

                # Helper Classification
                if self.helper_enabled and self.classifier and frame_count % 10 == 0:
                    try:
                        cid, conf = self.classifier.classify(frame)
                        dpg.set_value("lbl_helper_pred", f"Class ID: {cid} ({conf:.2f})")
                    except Exception as e:
                        pass

                # Draw
                display_frame = cv2.resize(frame, (WIDTH, HEIGHT))
                for res in last_results:
                    orig_h, orig_w = frame.shape[:2]
                    scale_x = WIDTH / orig_w
                    scale_y = HEIGHT / orig_h
                    (sx, sy, ex, ey) = res["box"]
                    sx = int(sx * scale_x); ex = int(ex * scale_x)
                    sy = int(sy * scale_y); ey = int(ey * scale_y)
                    color = res["color"]
                    cv2.rectangle(display_frame, (sx, sy), (ex, ey), color, 2)
                    cv2.putText(display_frame, res["display_label"], (sx, sy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                
                # Tracking
                if self.tracking_enabled and last_results:
                    # Convert results to format [sx, sy, ex, ey]
                    rects = []
                    for res in last_results:
                         rects.append(res["box"]) # Already (sx, sy, ex, ey)
                    
                    objects = self.tracker.update(rects)
                    
                    # Draw IDs
                    for (objectID, centroid) in objects.items():
                        # Scale centroid back to display size
                        orig_h, orig_w = frame.shape[:2]
                        scale_x = WIDTH / orig_w
                        scale_y = HEIGHT / orig_h
                        
                        cx = int(centroid[0] * scale_x)
                        cy = int(centroid[1] * scale_y)
                        
                        text = f"ID {objectID}"
                        cv2.putText(display_frame, text, (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(display_frame, (cx, cy), 4, (0, 255, 0), -1)

                # Update Texture
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
                display_frame = display_frame.astype(np.float32) / 255.0
                dpg.set_value(TEXTURE_TAG, display_frame.flatten())
                
                # Update FPS
                if self.camera:
                    current_fps = self.camera.get_fps()
                    self.fps_history.pop(0)
                    self.fps_history.append(current_fps)
                    if dpg.does_item_exist("fps_series"):
                        dpg.set_value("fps_series", [list(range(100)), self.fps_history])
            
            # Process Log Queue
            while not self.log_queue.empty():
                try:
                    msg = self.log_queue.get_nowait()
                    # Optional: Add to GUI log window
                    # dpg.add_text(msg, parent="log_list") 
                    # Note: Adding text in loop can lag if too many logs. 
                    # Better to just rely on the 'GuiLogHandler' if we connect it to a dpg logging widget or similar.
                    # For now just cleaning queue.
                except queue.Empty:
                    break

            if self.snapshot_message or dpg.get_value("lbl_snapshot_status"):
                 if time.time() - self.snapshot_timer > 2.0:
                     dpg.set_value("lbl_snapshot_status", "")

            dpg.render_dearpygui_frame()

        self.cleanup()

    def on_update_found(self, update_info):
        if update_info:
             # Use a small popup or text to notify
             # Since DPG is not thread-safe for direct UI creation from thread,
             # we queue a message or just set a flag.
             # For simplicity here, we'll try to set a global tag if it exists, or log it.
             # Ideally, we should add an "Update Available" button in the Top Bar.
             self.logger.info(f"UPDATE AVAILABLE: {update_info['version']}")
             self.log_queue.put(f"UPDATE AVAILABLE: {update_info['version']}")

    def cleanup(self):
        if self.camera: self.camera.stop()
        dpg.destroy_context()

if __name__ == "__main__":
    # Start with default
    app = ZertamApp("models/MobileNetSSD_deploy.caffemodel", "models/MobileNetSSD_deploy.prototxt")
    app.run()
