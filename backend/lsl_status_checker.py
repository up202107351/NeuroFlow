from PyQt5 import QtCore
import pylsl 
import time  

class LSLStatusChecker(QtCore.QObject):
    status_update = QtCore.pyqtSignal(bool, str) # connected (bool), message (str)
    finished = QtCore.pyqtSignal()

    def __init__(self, stream_type='EEG', resolve_timeout=1, check_interval=3): # Short timeout for check
        super().__init__()
        self.stream_type = stream_type
        self.resolve_timeout = resolve_timeout
        self.check_interval = check_interval # Seconds between checks
        self._running = False

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        print("LSLStatusChecker: Thread started.")
        while self._running:
            try:
                streams = pylsl.resolve_byprop('type', self.stream_type, 1, timeout=self.resolve_timeout)
                if streams:
                    # To be more robust, you could even try to open an inlet briefly
                    # inlet = pylsl.StreamInlet(streams[0], max_chunklen=1, max_buffered=1, processing_flags=pylsl.proc_dejitter)
                    # inlet.open_stream(timeout=0.5) # Try to open with short timeout
                    # inlet.close_stream()
                    self.status_update.emit(True, "Connected")
                else:
                    self.status_update.emit(False, "Not Detected")
            except Exception as e:
                print(f"LSLStatusChecker: Error during LSL check: {e}")
                self.status_update.emit(False, "Error Checking")

            # Wait for the check_interval, but break early if stop is requested
            for _ in range(int(self.check_interval / 0.1)): # Check stop flag every 0.1s
                if not self._running:
                    break
                time.sleep(0.1)
            if not self._running: # Ensure loop breaks if stop() was called during sleep
                break

        print("LSLStatusChecker: Thread finished.")
        self.finished.emit()

    def stop(self):
        print("LSLStatusChecker: Stop requested.")
        self._running = False
