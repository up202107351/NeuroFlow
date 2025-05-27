#!/usr/bin/env python3
"""
EEG Prediction Subscriber - ZMQ Client for EEG-based Feedback

This script connects to the EEG backend processor and receives mental state
predictions to be displayed in the user interface.
"""

import time
import zmq
import logging
import numpy as np
from PyQt5 import QtCore
import queue

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EEG_Subscriber')

class EEGPredictionSubscriber(QtCore.QObject):
    # Signals
    new_prediction_received = QtCore.pyqtSignal(dict)  # Emit the raw dict
    subscriber_error = QtCore.pyqtSignal(str)
    connection_status = QtCore.pyqtSignal(str)  # "Connecting", "Connected", "Disconnected"
    calibration_progress = QtCore.pyqtSignal(float)  # 0.0 to 1.0
    calibration_status = QtCore.pyqtSignal(str, dict)  # Status, data
    finished = QtCore.pyqtSignal()
    
    # Constants
    HEARTBEAT_TIMEOUT = 3.0  # Seconds to wait before considering connection lost
    
    def __init__(self, zmq_sub_address="tcp://localhost:5556", zmq_command_address="tcp://localhost:5558", zmq_sync_address="tcp://localhost:5557"):
        super().__init__()
        self.zmq_sub_address = zmq_sub_address
        self.zmq_command_address = zmq_command_address
        self.zmq_sync_address = zmq_sync_address
        self._running = False
        self.context = None
        self.subscriber = None
        self.command_socket = None
        self.sync_socket = None
        self.last_heartbeat_time = 0
        self.prediction_history = []
        self.max_history_size = 10
        self.session_active = False
        self.current_session_type = None
        self.command_queue = queue.Queue()  # Queue for non-blocking commands
        self.command_timer = None  # Timer for processing commands
        
    def send_command(self, command_dict):
        """Send a command to the EEG backend processor"""
        if not self.command_socket:
            self.connect_command_socket()
            
        if not self.command_socket:
            logger.error("Cannot send command: Command socket not connected")
            return {"status": "ERROR", "message": "Command socket not connected"}
            
        try:
            self.command_socket.send_json(command_dict)
            response = self.command_socket.recv_json()
            logger.info(f"Command response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return {"status": "ERROR", "message": f"Error: {str(e)}"}
            
    def connect_command_socket(self):
        """Connect to the command socket"""
        if self.command_socket:
            try:
                self.command_socket.close()
            except:
                pass
                
        try:
            self.command_socket = self.context.socket(zmq.REQ)
            self.command_socket.setsockopt(zmq.LINGER, 1000)  # Allow 1s for socket to close
            self.command_socket.connect(self.zmq_command_address)
            logger.info(f"Connected to command socket at {self.zmq_command_address}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to command socket: {e}")
            self.command_socket = None
            return False
            
    def synchronize_with_publisher(self):
        """Synchronize with the publisher using the sync socket"""
        try:
            # Create synchronization socket
            self.sync_socket = self.context.socket(zmq.REQ)
            self.sync_socket.connect(self.zmq_sync_address)
            
            # Send synchronization request
            self.sync_socket.send_string("SYNC")
            
            # Wait for response with timeout
            poller = zmq.Poller()
            poller.register(self.sync_socket, zmq.POLLIN)
            
            if poller.poll(5000):  # 5 second timeout
                response = self.sync_socket.recv_string()
                logger.info(f"Synchronization response: {response}")
                self.sync_socket.close()
                self.sync_socket = None
                return True
            else:
                logger.warning("Synchronization timeout")
                self.sync_socket.close()
                self.sync_socket = None
                return False
                
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")
            if self.sync_socket:
                self.sync_socket.close()
                self.sync_socket = None
            return False
            
    # UPDATED: Make this non-blocking by using the command queue
    def start_relaxation_session(self):
        """Start a relaxation session - non-blocking version"""
        if self.session_active:
            print("Session already active, not starting new one")
            logger.warning("Session already active")
            return False
                
        try:
            # Queue the command instead of sending directly
            cmd = {
                "command_type": "START_SESSION",
                "params": {"command": "START_SESSION", "session_type": "RELAXATION"}
            }
            print(f"Queuing command: {cmd}")
            self.command_queue.put(cmd)
            print(f"Command queue size now: {self.command_queue.qsize()}")
            logger.info("Relaxation session command queued")
            return True
        except Exception as e:
            print(f"Error queuing relaxation session command: {e}")
            logger.error(f"Error queuing relaxation session command: {e}")
            return False
            
    # UPDATED: Make this non-blocking by using the command queue  
    def start_focus_session(self):
        """Start a focus session - non-blocking version"""
        if self.session_active:
            logger.warning("Session already active")
            return False
            
        try:
            # Queue the command instead of sending directly
            self.command_queue.put({
                "command_type": "START_SESSION",
                "params": {"command": "START_SESSION", "session_type": "FOCUS"}
            })
            logger.info("Focus session command queued")
            return True
        except Exception as e:
            logger.error(f"Error queuing focus session command: {e}")
            return False
            
    # UPDATED: Make this non-blocking by using the command queue
    def stop_session(self):
        """Stop the current session - non-blocking version"""
        if not self.session_active:
            logger.warning("No active session to stop")
            return True
            
        try:
            # Queue the command instead of sending directly
            self.command_queue.put({
                "command_type": "STOP_SESSION",
                "params": {"command": "STOP_SESSION"}
            })
            logger.info("Stop session command queued")
            return True
        except Exception as e:
            logger.error(f"Error queuing stop session command: {e}")
            return False
    
    # NEW: Add a method to process the command queue
    def process_command_queue(self):
        """Process pending commands from the queue in a non-blocking way"""
        try:
            if not self.command_queue.empty():
                print(f"Command queue has {self.command_queue.qsize()} commands to process")
                
            while not self.command_queue.empty():
                cmd_data = self.command_queue.get_nowait()
                cmd_type = cmd_data.get("command_type")
                params = cmd_data.get("params", {})
                
                print(f"Processing queued command: {cmd_type} with params {params}")
                logger.info(f"Processing queued command: {cmd_type}")
                
                # Handle different types of commands
                if cmd_type == "START_SESSION":
                    # Make sure command socket is connected 
                    if not self.command_socket:
                        print("Command socket not connected, reconnecting...")
                        self.connect_command_socket()
                    
                    if not self.command_socket:
                        print("Failed to connect command socket!")
                        logger.error("Cannot send command: Command socket not connected")
                        continue
                    
                    # Send command with timeout
                    try:
                        print(f"Sending command to backend: {params}")
                        self.command_socket.send_json(params)
                        
                        # Wait for response with timeout
                        poller = zmq.Poller()
                        poller.register(self.command_socket, zmq.POLLIN)
                        
                        if poller.poll(2000):  # 2 second timeout
                            response = self.command_socket.recv_json()
                            print(f"Received command response: {response}")
                            logger.info(f"Command response: {response}")
                            
                            if response.get("status") == "SUCCESS":
                                self.session_active = True
                                self.current_session_type = params.get("session_type")
                                print(f"Session {self.current_session_type} activated successfully")
                        else:
                            print("Command response timeout!")
                            logger.warning("Command response timeout")
                            # Don't mark as session_active yet
                            
                    except Exception as e:
                        print(f"Error sending command: {e}")
                        logger.error(f"Error sending command: {e}")
                        
                elif cmd_type == "STOP_SESSION":
                    # Make sure command socket is connected 
                    if not self.command_socket:
                        print("Command socket not connected, reconnecting...")
                        self.connect_command_socket()
                    
                    if not self.command_socket:
                        print("Failed to connect command socket!")
                        logger.error("Cannot send command: Command socket not connected")
                        continue
                    
                    # Send command with timeout
                    try:
                        print(f"Sending command to backend: {params}")
                        self.command_socket.send_json(params)
                        
                        # Wait for response with timeout
                        poller = zmq.Poller()
                        poller.register(self.command_socket, zmq.POLLIN)
                        
                        if poller.poll(2000):  # 2 second timeout
                            response = self.command_socket.recv_json()
                            print(f"Received command response: {response}")
                            logger.info(f"Command response: {response}")
                            
                            if response.get("status") == "SUCCESS":
                                print(f"Session {self.current_session_type} deactivated successfully")
                                self.session_active = False
                                self.current_session_type = None
                        else:
                            print("Command response timeout!")
                            logger.warning("Command response timeout")
                            # Don't mark as session_active yet
                            
                    except Exception as e:
                        print(f"Error sending command: {e}")
                        logger.error(f"Error sending command: {e}")
                
                # Mark command as done
                self.command_queue.task_done()
                
        except queue.Empty:
            pass  # Queue is empty, nothing to do
        except Exception as e:
            logger.error(f"Error processing command queue: {e}")
            self.subscriber_error.emit(f"Command processing error: {e}")
    
    @QtCore.pyqtSlot()
    def run(self):
        """Main subscriber loop - runs in a separate thread"""
        self._running = True
        self.context = zmq.Context()
        
        # Connect command socket
        print("Attempting to connect command socket...")
        self.connect_command_socket()
        
        # Connect subscriber socket
        self.subscriber = self.context.socket(zmq.SUB)
        print(f"Attempting to connect to ZMQ PUB at {self.zmq_sub_address}")
        self.connection_status.emit(f"Connecting to {self.zmq_sub_address}...")
        
        try:
            self.subscriber.connect(self.zmq_sub_address)
            self.subscriber.subscribe("")  # Subscribe to all messages
            
            # Synchronize with publisher
            print("Attempting to synchronize with publisher...")
            sync_result = self.synchronize_with_publisher()
            print(f"Synchronization result: {sync_result}")
            
            self.connection_status.emit(f"Connected to EEG Backend.")
            print(f"Connected to ZMQ Publisher.")
            self.last_heartbeat_time = time.time()
            
            # VERY IMPORTANT - Create command processing timer in the thread
            print("Setting up command processing timer...")
            self.command_timer = QtCore.QTimer()
            self.command_timer.timeout.connect(self.process_command_queue)
            self.command_timer.start(100)  # Process commands every 100ms
            print("Command timer started!")
            
        except zmq.error.ZMQError as e:
            err_msg = f"ZMQ connection error: {e}"
            print(err_msg)
            self.subscriber_error.emit(err_msg)
            self.connection_status.emit(f"Connection Failed: {e}")
            self._running = False
                
        # Set up poller for subscriber socket
        poller = zmq.Poller()
        poller.register(self.subscriber, zmq.POLLIN)
        
        # Main subscriber loop
        while self._running:
            try:
                # Poll with a timeout to allow checking _running flag
                socks = dict(poller.poll(timeout=500))  # 500ms timeout
                
                # Add explicit command queue processing here too, just to be sure
                if self._running:  # Check again after polling
                    self.process_command_queue()
                
                # Check for heartbeat timeout
                current_time = time.time()
                if current_time - self.last_heartbeat_time > self.HEARTBEAT_TIMEOUT:
                    logger.warning("Heartbeat timeout - connection may be lost")
                    self.connection_status.emit("Connection lost - no heartbeat")
                    # Don't stop running, just report the potential issue
                
                # Process incoming messages
                if self.subscriber in socks and socks[self.subscriber] == zmq.POLLIN:
                    message_json = self.subscriber.recv_json()
                    
                    # Handle different message types
                    if "message_type" in message_json:
                        if message_json["message_type"] == "HEARTBEAT":
                            # Update heartbeat time
                            self.last_heartbeat_time = current_time
                            
                        elif message_json["message_type"] == "CONNECTION_STATUS":
                            # Handle connection status updates
                            self.connection_status.emit(message_json["status"])
                            
                        elif message_json["message_type"] == "CALIBRATION_PROGRESS":
                            # Handle calibration progress updates
                            self.calibration_progress.emit(message_json["progress"])
                            
                        elif message_json["message_type"] == "CALIBRATION_STATUS":
                            # Handle calibration status updates
                            self.calibration_status.emit(
                                message_json["status"], 
                                message_json.get("baseline", {})
                            )
                            
                        elif message_json["message_type"] == "PREDICTION":
                            # Handle prediction updates
                            
                            # Add to history for smoothing if needed
                            self.prediction_history.append(message_json)
                            if len(self.prediction_history) > self.max_history_size:
                                self.prediction_history.pop(0)
                                
                            # Emit the raw prediction first
                            self.new_prediction_received.emit(message_json)
                            
                    else:
                        # Backward compatibility with old message format
                        self.new_prediction_received.emit(message_json)
                        
            except zmq.error.ContextTerminated:
                logger.info("ZMQ Context terminated, subscriber stopping.")
                self._running = False  # Exit loop if context is terminated
                
            except Exception as e:
                # Avoid flooding with errors if backend is down after initial connect
                if self._running:  # Only log if we are supposed to be running
                    logger.error(f"Error receiving/parsing ZMQ message: {e}")
                    self.subscriber_error.emit(f"ZMQ receive error: {e}")
                time.sleep(0.1)  # Small pause
                
        # --- Cleanup when loop exits ---
        self.cleanup()
        
        logger.info("EEGPredictionSubscriber run loop finished.")
        self.connection_status.emit("Disconnected from EEG Backend.")
        self.finished.emit()
        
    def stop(self):
        """Stop the subscriber"""
        logger.info("EEGPredictionSubscriber stop requested.")
        
        # First try to stop any active session
        if self.session_active:
            self.stop_session()
            
        # Stop the command timer
        if self.command_timer and self.command_timer.isActive():
            self.command_timer.stop()
            
        # Then stop the subscriber
        self._running = False
        
    def cleanup(self):
        """Clean up resources"""
        # Stop the command timer
        if self.command_timer and self.command_timer.isActive():
            self.command_timer.stop()
            
        if self.subscriber:
            self.subscriber.close()
            
        if self.command_socket:
            # Try to send shutdown command
            try:
                self.command_socket.send_json({"command": "SHUTDOWN"})
                # Wait briefly for response
                poller = zmq.Poller()
                poller.register(self.command_socket, zmq.POLLIN)
                if poller.poll(1000):  # 1 second timeout
                    _ = self.command_socket.recv_json()
            except:
                pass
            self.command_socket.close()
            
        if self.sync_socket:
            self.sync_socket.close()
            
        if self.context and not self.context.closed:
            try:
                self.context.term()
            except zmq.error.ZMQError as e:
                logger.error(f"Error terminating ZMQ context: {e}")