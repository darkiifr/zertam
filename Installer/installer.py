import os
import shutil
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

def install():
    # 1. Define Source Files
    if getattr(sys, 'frozen', False):
        # Bundled mode: Source is in temp _MEIPASS
        base_dir = sys._MEIPASS
        exe_source = os.path.join(base_dir, "ZertamVision.exe")
        # Docs also bundled in root of bundle
        doc_root = base_dir 
    else:
        # Dev mode: Source is in ../dist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(base_dir) 
        exe_source = os.path.join(root_dir, "dist", "ZertamVision.exe")
        doc_root = root_dir

    docs_to_copy = ["LICENSE", "SECURITY.md", "README.md"]
    
    if not os.path.exists(exe_source):
        messagebox.showerror("Error", f"Could not find source executable at:\n{exe_source}\n\nPlease build the project first.")
        return

    # 2. Ask User for Install Directory & Options
    root = tk.Tk()
    root.withdraw()
    
    # Simple Dialog for Options
    dialog = tk.Toplevel(root)
    dialog.title("Install Options")
    dialog.geometry("300x150")
    
    var_shortcut = tk.BooleanVar(value=True)
    tk.Checkbutton(dialog, text="Create Desktop Shortcut", variable=var_shortcut).pack(pady=20)
    
    def on_ok():
        dialog.destroy()
        
    tk.Button(dialog, text="Select Install Dir & Continue", command=on_ok).pack()
    root.wait_window(dialog)
    
    install_dir = filedialog.askdirectory(title="Select Installation Directory")
    
    if not install_dir:
        return # User cancelled

    target_dir = os.path.join(install_dir, "ZertamVision")
    
    try:
        # 3. Create Directory
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            
        # 4. Copy Executable
        print(f"Copying EXE to {target_dir}...")
        shutil.copy2(exe_source, target_dir)
        
        # 5. Copy Docs
        for doc in docs_to_copy:
            src = os.path.join(doc_root, doc)
            if os.path.exists(src):
                print(f"Copying {doc}...")
                shutil.copy2(src, target_dir)
                
        # 6. Create Desktop Shortcut (Windows Only)
        if var_shortcut.get():
            try:
                import winshell
                from win32com.client import Dispatch
                
                desktop = winshell.desktop()
                path = os.path.join(desktop, "Zertam Vision.lnk")
                target = os.path.join(target_dir, "ZertamVision.exe")
                
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(path)
                shortcut.TargetPath = target
                shortcut.WorkingDirectory = target_dir
                shortcut.IconLocation = target
                shortcut.save()
                print("Desktop shortcut created.")
            except ImportError:
                print("winshell/pywin32 not found. Skipping shortcut creation.")
            except Exception as e:
                print(f"Failed to create shortcut: {e}")

        messagebox.showinfo("Success", f"Zertam Vision installed successfully to:\n{target_dir}")
        
    except Exception as e:
        messagebox.showerror("Installation Failed", str(e))

if __name__ == "__main__":
    print("Launching Installer...")
    install()
