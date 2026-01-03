import json
import urllib.request
import logging
from threading import Thread
import re

logger = logging.getLogger("Updater")

CURRENT_VERSION = "2.0.0"
REPO_URL = "https://api.github.com/repos/darkiifr/zertam/releases/latest"

def parse_version(v_str):
    """
    Parses 'v1.2.3-beta' into (1, 2, 3, 'beta').
    Simple regex parser.
    """
    v_str = v_str.lower().strip().lstrip('v')
    # Match major.minor.patch[-suffix]
    pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9]+))?$"
    match = re.match(pattern, v_str)
    if match:
        major, minor, patch, suffix = match.groups()
        return (int(major), int(minor), int(patch), suffix if suffix else "z_final") 
        # z_final > beta for string comparison (e.g. 'z' > 'b') works simply
    return (0, 0, 0, "")

def is_newer(remote, local):
    """
    Returns True if remote tuple > local tuple.
    Tuple: (major, minor, patch, suffix)
    """
    # Compare numbers
    if remote[:3] > local[:3]: return True
    if remote[:3] < local[:3]: return False
    
    # Compare suffix (lexicographical check for simplicity)
    # local=final(z_final), remote=beta -> 'beta' < 'z_final' -> Not Newer
    # local=beta, remote=final(z_final) -> 'z_final' > 'beta' -> Newer
    return remote[3] > local[3]

def check_for_updates(callback):
    def _check():
        try:
            logger.info("Checking for updates...")
            try:
                with urllib.request.urlopen(REPO_URL) as response:
                    if response.status == 200:
                        data = json.loads(response.read().decode())
                        tag_name = data.get("tag_name", "")
                        
                        remote_ver = parse_version(tag_name)
                        local_ver = parse_version(CURRENT_VERSION)
                        
                        if is_newer(remote_ver, local_ver):
                            logger.info(f"New version found: {tag_name}")
                            callback({
                                "version": tag_name,
                                "url": data.get("html_url", ""),
                                "body": data.get("body", "")
                            })
                            return
            except urllib.error.HTTPError as e:
                logger.warning(f"Update check failed (HTTP): {e}")
            except Exception as e:
                logger.error(f"Update check failed: {e}")
        except Exception as e:
            logger.error(f"Update thread error: {e}")
            callback(None)
        except Exception as e:
            logger.error(f"Update check failed: {e}")
            callback(None)

    t = Thread(target=_check, daemon=True)
    t.start()
