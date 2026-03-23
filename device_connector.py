"""
device_connector.py — ADB / UIAutomator2 connection, screen capture, and touch simulation.

Uses uiautomator2 for screen capture (fast, in-memory PIL images) and
raw `adb shell input tap` for touch injection (more reliable than u2's
click on some devices).
"""

import subprocess
import numpy as np
import uiautomator2 as u2
import cv2
import time


class DeviceConnector:
    """Manage the connection to a physical Android device over USB."""

    def __init__(self, device_serial: str | None = None):
        """
        Connect to the Android device.

        Parameters
        ----------
        device_serial : str or None
            ADB serial number of the target device.
            If None, uiautomator2 will auto-detect the first USB device.
        """
        print("[DeviceConnector] Connecting to device …")
        self._serial = device_serial

        # UIAutomator2 — used for screenshots
        if device_serial:
            self.device = u2.connect(device_serial)
        else:
            self.device = u2.connect()  # auto-detect

        info = self.device.info
        display = self.device.window_size()
        self.screen_width = display[0]
        self.screen_height = display[1]

        # Build the ADB command prefix (handles serial if provided)
        self._adb_prefix = ["adb"]
        if device_serial:
            self._adb_prefix += ["-s", device_serial]

        print(f"[DeviceConnector] Connected: {info.get('productName', 'Unknown')}")
        print(f"[DeviceConnector] Screen size: {self.screen_width}×{self.screen_height}")

    # ------------------------------------------------------------------
    # Screen capture  (via uiautomator2 — works reliably)
    # ------------------------------------------------------------------

    def capture_screen(self) -> np.ndarray:
        """
        Capture the current screen and return it as a BGR NumPy array.

        Returns
        -------
        np.ndarray
            Screenshot in BGR colour space (H × W × 3), same as OpenCV.
        """
        pil_image = self.device.screenshot()
        rgb_array = np.array(pil_image)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array

    # ------------------------------------------------------------------
    # Touch helpers  (via ADB shell — universally reliable)
    # ------------------------------------------------------------------

    def tap(self, x: int, y: int, delay: float = 0.0) -> None:
        """
        Send a single tap event to the device at (x, y)
        using ``adb shell input tap``.

        Parameters
        ----------
        x, y  : int
            Absolute screen coordinates.
        delay : float
            Seconds to wait *after* the tap.
        """
        cmd = self._adb_prefix + ["shell", "input", "tap", str(x), str(y)]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if delay > 0:
            time.sleep(delay)

    def tap_sequence(self, actions: list[tuple[int, int] | float | int]) -> None:
        """
        Execute a sequence of taps using a single ADB shell command.
        
        This drastically reduces USB latency by concatenating commands
        with semicolons (e.g. `input tap x1 y1 ; sleep 0.5 ; input tap x2 y2`).

        Parameters
        ----------
        actions : list of (x, y) tuples or numeric sleep delays (seconds)
        """
        if not actions:
            return
            
        parts = []
        for action in actions:
            if isinstance(action, (float, int)):
                parts.append(f"sleep {action}")
            else:
                parts.append(f"input tap {action[0]} {action[1]}")
                
        shell_cmd = " ; ".join(parts)
        
        cmd = self._adb_prefix + ["shell", shell_cmd]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the device screen."""
        return self.screen_width, self.screen_height
