import logging
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from cva_runtime.control_plane.desktop_adapter import DesktopAdapter

# Configure logging to see the output
logging.basicConfig(level=logging.INFO)

def test_desktop_adapter():
    print("Testing Desktop Adapter...")
    adapter = DesktopAdapter()
    
    windows = adapter.get_windows()
    print(f"Detected {len(windows)} windows.")
    
    for win in windows[:5]: # Show first 5
        print(f"  > {win['id']}: {win['title']}")
        details = adapter.get_window_details(win['id'])
        if details:
            print(f"    Geometry: {details['width']}x{details['height']} at ({details['x']}, {details['y']})")

    # Try root screenshot
    success = adapter.take_screenshot("scratch/root_test.xwd")
    if success:
        print("✓ Root screenshot captured (XWD format).")
    else:
        print("✗ Screenshot failed.")

if __name__ == "__main__":
    test_desktop_adapter()
