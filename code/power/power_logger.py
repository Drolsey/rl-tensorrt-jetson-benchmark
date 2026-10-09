import subprocess
import threading
import re
import time
import numpy as np


class PowerLogger:
    """
    Reads tegrastats on a background thread between start() and stop()
    and keeps every power sample (in mW) for summary statistics.
    """

    def __init__(self):
        self.running = False
        self.samples = []

    def _run(self):
        process = subprocess.Popen(
            ["tegrastats"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        while self.running:
            line = process.stdout.readline()

            # take the first mW reading that follows the GR3D_FREQ
            # (GPU load) field; which rail that is depends on the
            # tegrastats output layout of the board
            match = re.search(r'GR3D_FREQ.*?([0-9]+)%.*?([0-9]+)mW', line)

            if match:
                self.samples.append(float(match.group(2)))

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()

    def stats(self):
        return {
            "power_mean_mW": float(np.mean(self.samples)) if self.samples else 0.0,
            "power_std_mW": float(np.std(self.samples)) if self.samples else 0.0
        }
