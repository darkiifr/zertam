import platform
import psutil
import subprocess

def get_cpu_name():
    try:
        if platform.system() == "Windows":
            return platform.processor()
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":")[1].strip()
        return platform.processor()
    except:
        return "Unknown CPU"

def get_gpu_name():
    try:
        if platform.system() == "Windows":
            # Uses wmic to get GPU name
            cmd = "wmic path win32_VideoController get name"
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            if out:
                lines = out.decode('utf-8').strip().split('\n')
                if len(lines) > 1:
                    return lines[1].strip() # Return first GPU found
        # Linux implementation omitted for now as user is on Windows
    except:
        pass
    return "Unknown/Integrated GPU"

def get_system_stats():
    """Returns a dict with cpu_percent, ram_percent"""
    return {
        "cpu": psutil.cpu_percent(interval=None),
        "ram": psutil.virtual_memory().percent
    }
