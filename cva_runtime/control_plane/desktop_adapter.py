import subprocess
import logging
import os
import re

log = logging.getLogger("DesktopAdapter")

class DesktopAdapter:
    """
    Low-level adapter for Linux X11 desktop automation.
    Uses standard X11 utilities to discover and interact with windows.
    """
    
    def __init__(self, display=":1"):
        self.display = display
        self.env = os.environ.copy()
        self.env["DISPLAY"] = self.display

    def get_windows(self):
        """Returns a list of open windows using xwininfo."""
        try:
            # Get all window IDs
            res = subprocess.run(
                ["xwininfo", "-root", "-children"], 
                env=self.env, 
                capture_output=True, 
                text=True
            )
            if res.returncode != 0:
                return []

            windows = []
            # Parse output:  0x123456 "Window Title": ("class" "name")  100x100+0+0  +0+0
            pattern = re.compile(r'(0x[0-9a-fA-F]+)\s+"([^"]*)":')
            for line in res.stdout.splitlines():
                match = pattern.search(line)
                if match:
                    windows.append({
                        "id": match.group(1),
                        "title": match.group(2)
                    })
            return windows
        except Exception as e:
            log.error(f"Failed to get windows: {e}")
            return []

    def get_window_details(self, window_id):
        """Returns geometry and state for a specific window."""
        try:
            res = subprocess.run(
                ["xwininfo", "-id", window_id], 
                env=self.env, 
                capture_output=True, 
                text=True
            )
            if res.returncode != 0:
                return None
            
            # Extract geometry
            geometry = {}
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith("-geometry"):
                    geometry["raw"] = line.split()[-1]
                elif "Width:" in line:
                    geometry["width"] = int(line.split()[-1])
                elif "Height:" in line:
                    geometry["height"] = int(line.split()[-1])
                elif "Absolute upper-left X:" in line:
                    geometry["x"] = int(line.split()[-1])
                elif "Absolute upper-left Y:" in line:
                    geometry["y"] = int(line.split()[-1])
            
            return geometry
        except Exception as e:
            log.error(f"Failed to get window details for {window_id}: {e}")
            return None

    def take_screenshot(self, output_path="screenshot.xwd", window_id=None):
        """Takes a raw XWD screenshot of the root window or a specific window."""
        try:
            cmd = ["xwd", "-display", self.display, "-out", output_path]
            if window_id:
                cmd.extend(["-id", window_id])
            else:
                cmd.append("-root")
            
            res = subprocess.run(cmd, env=self.env, capture_output=True, text=True)
            if res.returncode == 0:
                log.info(f"Screenshot saved to {output_path}")
                return True
            else:
                log.error(f"Screenshot failed (rc={res.returncode}): {res.stderr}")
                return False
        except Exception as e:
            log.error(f"Screenshot exception: {e}")
            return False
