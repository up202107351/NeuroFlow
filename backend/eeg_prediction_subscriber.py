from PyQt5 import QtCore
import zmq
import time

class EEGPredictionSubscriber(QtCore.QObject):
    new_prediction_received = QtCore.pyqtSignal(dict) # Emit the raw dict
    subscriber_error = QtCore.pyqtSignal(str)
    connection_status = QtCore.pyqtSignal(str) # "Connecting", "Connected", "Disconnected"
    finished = QtCore.pyqtSignal()

    def __init__(self, zmq_sub_address="tcp://localhost:5556"):
        super().__init__()
        self.zmq_sub_address = zmq_sub_address
        self._running = False
        self.context = None
        self.subscriber = None

    @QtCore.pyqtSlot()
    def run(self):
        self._running = True
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        print(f"Frontend: Attempting to connect to ZMQ PUB at {self.zmq_sub_address}")
        self.connection_status.emit(f"Connecting to {self.zmq_sub_address}...")
        try:
            self.subscriber.connect(self.zmq_sub_address)
            self.subscriber.subscribe("") # Subscribe to all messages
            self.connection_status.emit(f"Connected to EEG Backend.")
            print(f"Frontend: Connected to ZMQ Publisher.")
        except zmq.error.ZMQError as e:
            err_msg = f"Frontend: ZMQ connection error: {e}"
            print(err_msg)
            self.subscriber_error.emit(err_msg)
            self.connection_status.emit(f"Connection Failed: {e}")
            self._running = False # Stop if connection fails initially

        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN)

        while self._running:
            try:
                # Poll with a timeout (e.g., 1000ms) to allow checking _running flag
                socks = dict(poller.poll(timeout=1000))
                if self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN:
                    message_json = self.subscriber.recv_json()
                    self.new_prediction_received.emit(message_json)
            except zmq.error.ContextTerminated:
                print("Frontend: ZMQ Context terminated, subscriber stopping.")
                self._running = False # Exit loop if context is terminated
            except Exception as e:
                # Avoid flooding with errors if backend is down after initial connect
                if self._running: # Only log if we are supposed to be running
                    print(f"Frontend: Error receiving/parsing ZMQ message: {e}")
                    self.subscriber_error.emit(f"ZMQ receive error: {e}")
                    # Optionally emit a disconnected status here if errors persist
                time.sleep(0.1) # Small pause

        # --- Ensure 'finished' is emitted when the loop exits ---
        if self.subscriber:
            self.subscriber.close()
        if self.context:
            # Be careful with term() if other sockets from this context might still be in use
            # If this is the only user of self.context, then it's okay.
            if not self.context.closed: # Check if not already closed
                try:
                    self.context.term()
                except zmq.error.ZMQError as e:
                    print(f"Frontend: Error terminating ZMQ context: {e}")

        print("Frontend: EEGPredictionSubscriber run loop finished.")
        self.connection_status.emit("Disconnected from EEG Backend.") # Update status
        self.finished.emit() # <--- EMIT THE SIGNAL HERE

    def stop(self):
        print("Frontend: EEGPredictionSubscriber stop requested.")
        self._running = False
        # Note: The 'finished' signal will be emitted from the run() method
        # when its loop naturally exits due to _running being False.