#!/usr/bin/env python3
"""
ZMQ Port Cleanup Utility

This script forcibly cleans up ZMQ ports that might be in use from crashed processes.
Run this before starting your application if you encounter "Address in use" errors.
"""

import os
import sys
import zmq
import time
import logging
import signal
import subprocess
import psutil  # Make sure to install this: pip install psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ZMQ_Cleanup')

# Default ports used by your application
DEFAULT_PORTS = [5556, 5557, 5558]  # Publisher, Sync, Command ports

def find_process_using_port(port):
    """Find the process ID using a specific port"""
    if os.name == 'nt':  # Windows
        try:
            # Use netstat on Windows
            output = subprocess.check_output(f'netstat -ano | findstr ":{port}"', shell=True).decode()
            if output:
                lines = output.strip().split('\n')
                for line in lines:
                    if f":{port}" in line and "LISTENING" in line:
                        parts = line.strip().split()
                        pid = parts[-1]
                        return int(pid)
        except subprocess.CalledProcessError:
            pass
    else:  # Linux/Mac
        try:
            # Use lsof on Linux/Mac
            output = subprocess.check_output(f'lsof -i :{port} -t', shell=True).decode()
            if output:
                return int(output.strip())
        except subprocess.CalledProcessError:
            pass
    return None

def kill_process(pid):
    """Kill a process by PID"""
    try:
        process = psutil.Process(pid)
        process_name = process.name()
        
        logger.info(f"Killing process {pid} ({process_name})")
        if os.name == 'nt':  # Windows
            subprocess.call(['taskkill', '/F', '/PID', str(pid)])
        else:  # Linux/Mac
            os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)  # Give it a moment
        return True
    except psutil.NoSuchProcess:
        logger.warning(f"Process {pid} not found")
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
    return False

def test_port(port):
    """Test if a port is in use by trying to bind to it"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    try:
        socket.bind(f"tcp://*:{port}")
        socket.unbind(f"tcp://*:{port}")
        socket.close()
        context.term()
        return False  # Port is free
    except zmq.error.ZMQError:
        socket.close()
        context.term()
        return True  # Port is in use

def cleanup_zmq_port(port):
    """Cleanup a specific ZMQ port"""
    logger.info(f"Checking port {port}...")
    
    # First check if port is actually in use
    if not test_port(port):
        logger.info(f"Port {port} is already free")
        return True
        
    # Find the process using the port
    pid = find_process_using_port(port)
    if pid:
        logger.info(f"Found process {pid} using port {port}")
        # Kill the process
        if kill_process(pid):
            # Check if port is now free
            if not test_port(port):
                logger.info(f"Successfully freed port {port}")
                return True
            else:
                logger.warning(f"Port {port} still in use after killing process {pid}")
                return False
        else:
            logger.warning(f"Failed to kill process {pid}")
            return False
    else:
        logger.warning(f"No process found using port {port}, but it appears to be in use")
        
        # Try the ZMQ context reset approach
        try:
            # Create a new context and try to reset
            logger.info("Attempting ZMQ context reset...")
            ctx = zmq.Context()
            ctx.term()
            time.sleep(1)
            
            # Check if it worked
            if not test_port(port):
                logger.info(f"Successfully freed port {port} through context reset")
                return True
        except Exception as e:
            logger.error(f"Error during ZMQ context reset: {e}")
            
        return False

def cleanup_all_zmq_ports(ports=None):
    """Cleanup all specified ZMQ ports"""
    if ports is None:
        ports = DEFAULT_PORTS
        
    logger.info(f"Starting ZMQ port cleanup for ports: {ports}")
    
    success = True
    for port in ports:
        if not cleanup_zmq_port(port):
            success = False
            
    if success:
        logger.info("All ports successfully cleaned up")
    else:
        logger.warning("Some ports could not be cleaned up")
        
    return success

if __name__ == "__main__":
    ports_to_clean = DEFAULT_PORTS
    
    # Allow custom ports via command line
    if len(sys.argv) > 1:
        ports_to_clean = [int(p) for p in sys.argv[1:]]
        
    cleanup_all_zmq_ports(ports_to_clean)