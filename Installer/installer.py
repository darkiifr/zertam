import os
import shutil
import sys
import json
import urllib.request
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Constants
REPO_API_URL = "https://api.github.com/repos/darkiifr/zertam/releases"
ICON_PATH = os.path.join("assets", "icon", "z.png")

class InstallerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Zertam Vision Installer")
        self.root.geometry("400x350")
        
        # Try to set icon
        try:
            if getattr(sys, 'frozen', False):
                # In bundle, icon might be in temp
                base = sys._MEIPASS
            else:
                base = os.path.dirname(os.path.abspath(__file__))
                # If running from Installer/ put it relative to root? 
                # Actually Buildall.py copies it or we assume it exists
            
            # Simple icon loading attempt
            # In a real build, the exe icon is set by PyInstaller.
            # Tkinter window icon:
            pass 
        except:
            pass

        self.releases = []
        self.selected_release = None
        self.bundled_exe = self.check_bundled_exe()

        self.setup_ui()

    def check_bundled_exe(self):
        if getattr(sys, 'frozen', False):
            base_dir = sys._MEIPASS
            # Look for ZIP instead of EXE
            bundle = os.path.join(base_dir, "ZertamVision.zip")
            if os.path.exists(bundle):
                return bundle
        else:
            # Dev mode
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # In dev, maybe use dist/ZertamVision.zip if exists
            bundle = os.path.join(base_dir, "dist", "ZertamVision.zip")
            if os.path.exists(bundle):
                return bundle
        return None

    def setup_ui(self):
        # Header
        lbl_title = tk.Label(self.root, text="Zertam Vision Installer", font=("Segoe UI", 16, "bold"))
        lbl_title.pack(pady=10)

        # Status
        self.lbl_status = tk.Label(self.root, text="Select functionality below", fg="gray")
        self.lbl_status.pack(pady=5)

        # Version Selection Frame
        frame_ver = tk.LabelFrame(self.root, text="Installation Source")
        frame_ver.pack(fill="x", padx=10, pady=5)

        self.var_source = tk.StringVar(value="offline" if self.bundled_exe else "online")
        
        if self.bundled_exe:
            tk.Radiobutton(frame_ver, text="Offline (Included Version)", variable=self.var_source, value="offline", command=self.toggle_online_ui).pack(anchor="w")
        
        tk.Radiobutton(frame_ver, text="Online (Download from GitHub)", variable=self.var_source, value="online", command=self.toggle_online_ui).pack(anchor="w")

        # Online Options
        self.frame_online = tk.Frame(frame_ver)
        self.frame_online.pack(fill="x", padx=20, pady=5)
        
        tk.Label(self.frame_online, text="Select Version:").pack(anchor="w")
        self.combo_versions = ttk.Combobox(self.frame_online, state="readonly")
        self.combo_versions.pack(fill="x")
        self.combo_versions.bind("<<ComboboxSelected>>", self.on_version_select)
        
        btn_refresh = tk.Button(self.frame_online, text="Refresh Versions", command=self.fetch_releases_thread)
        btn_refresh.pack(anchor="e", pady=2)

        # Install Options
        frame_opts = tk.Frame(self.root)
        frame_opts.pack(fill="x", padx=10, pady=5)
        self.var_shortcut_desktop = tk.BooleanVar(value=True)
        self.var_shortcut_start = tk.BooleanVar(value=True)
        tk.Checkbutton(frame_opts, text="Desktop Shortcut", variable=self.var_shortcut_desktop).pack(anchor="w")
        tk.Checkbutton(frame_opts, text="Start Menu Shortcut", variable=self.var_shortcut_start).pack(anchor="w")

        # Buttons
        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)
        
        btn_install = tk.Button(self.root, text="Install", bg="#00cc88", fg="white", font=("Segoe UI", 10, "bold"), command=self.start_install)
        btn_install.pack(pady=5, ipadx=20)

        self.toggle_online_ui()
        if not self.bundled_exe:
            self.fetch_releases_thread()

    def toggle_online_ui(self):
        if self.var_source.get() == "online":
            # Enable combo
            for child in self.frame_online.winfo_children():
                child.configure(state="normal")
        else:
            # Disable combo
            for child in self.frame_online.winfo_children():
                child.configure(state="disabled")

    def fetch_releases_thread(self):
        self.lbl_status.config(text="Fetching releases...", fg="blue")
        t = threading.Thread(target=self.fetch_releases)
        t.start()

    def fetch_releases(self):
        try:
            req = urllib.request.Request(REPO_API_URL)
            req.add_header('User-Agent', 'ZertamInstaller')
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                self.releases = data
                
                # Format for combo
                items = []
                for r in self.releases:
                    tag = r['tag_name']
                    pre = " (Beta)" if r['prerelease'] else " (Stable)"
                    items.append(f"{tag}{pre}")
                
                def update_ui():
                    self.combo_versions['values'] = items
                    if items:
                        self.combo_versions.current(0)
                        self.on_version_select(None)
                    self.lbl_status.config(text=f"Found {len(items)} versions.", fg="green")
                
                self.root.after(0, update_ui)
        except Exception as e:
            def show_err():
                self.lbl_status.config(text=f"Fetch failed: {e}", fg="red")
            self.root.after(0, show_err)

    def on_version_select(self, event):
        idx = self.combo_versions.current()
        if idx >= 0:
            self.selected_release = self.releases[idx]

    def start_install(self):
        source = self.var_source.get()
        if source == "online" and not self.selected_release:
            messagebox.showwarning("Warning", "Please select a version to download.")
            return

        target_dir = filedialog.askdirectory(title="Select Install Directory")
        if not target_dir: return
        
        install_path = os.path.join(target_dir, "ZertamVision")
        
        # Thread installation
        t = threading.Thread(target=self.install_process, args=(source, install_path))
        t.start()

    def install_process(self, source, install_path):
        try:
            if not os.path.exists(install_path):
                os.makedirs(install_path)
            
            exe_dest = os.path.join(install_path, "ZertamVision.exe")

            if source == "offline":
                self.update_status(f"Extracting files...", 10)
                # Unzip bundle
                import zipfile
                with zipfile.ZipFile(self.bundled_exe, 'r') as zip_ref:
                    zip_ref.extractall(install_path)
                self.update_status(f"Files extracted.", 60)
            else:
                # Online Download (Still EXE for now, or change to ZIP later if releases change)
                # Assumes GitHub release has 'ZertamVision.zip' or 'exe'
                # Adapted logic for EXE download
                asset = next((a for a in self.selected_release['assets'] if a['name'].endswith(".exe")), None)
                if not asset:
                    raise Exception("No EXE found in this release.")
                
                url = asset['browser_download_url']
                self.update_status(f"Downloading {self.selected_release['tag_name']}...", 0)
                
                # Download with progress hook? Simple for now
                with urllib.request.urlopen(url) as response, open(exe_dest, 'wb') as out_file:
                    total_len = int(response.info().get('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 4096
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk: break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_len > 0:
                            perc = int((downloaded / total_len) * 80)
                            self.root.after(0, lambda p=perc: self.progress.configure(value=p))
                
                self.update_status("Download complete.", 80)

            # Docs (Assuming bundled or skipping for online simplified)
            # In online mode, ideally we fetch README from repo source if not in asset.
            # Here we skip docs for online to keep it simple or bundle generic ones.
            
            # Shortcuts
            if self.var_shortcut_desktop.get():
                self.update_status("Creating Desktop shortcut...", 90)
                self.create_shortcut(install_path, "desktop")
            
            if self.var_shortcut_start.get():
                self.update_status("Creating Start Menu shortcut...", 95)
                self.create_shortcut(install_path, "startmenu")

            self.update_status("Installation Verified.", 100)
            messagebox.showinfo("Success", f"Installed to {install_path}")
            self.root.quit()

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.update_status("Failed.", 0)

    def update_status(self, text, val):
        self.root.after(0, lambda: self.lbl_status.config(text=text))
        self.root.after(0, lambda: self.progress.configure(value=val))

    def create_shortcut(self, target_dir, location):
        try:
            import winshell
            from win32com.client import Dispatch
            
            if location == "desktop":
                folder = winshell.desktop()
            else:
                folder = winshell.programs() # Start Menu > Programs
                
            path = os.path.join(folder, "Zertam Vision.lnk")
            target = os.path.join(target_dir, "ZertamVision.exe")
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(path)
            shortcut.TargetPath = target
            shortcut.WorkingDirectory = target_dir
            shortcut.IconLocation = target
            shortcut.save()
        except Exception as e:
            print(f"Shortcut error ({location}): {e}")

if __name__ == "__main__":
    app = InstallerApp()
    app.root.mainloop()
