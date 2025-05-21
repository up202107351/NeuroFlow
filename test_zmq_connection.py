#!/usr/bin/env python3
"""
Simple ZMQ Connection Tester

Tests the connection between ZMQ publisher and subscriber
"""

import zmq
import time
import threading
import sys

def run_publisher():
    """Run a simple ZMQ publisher"""
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    
    # Bind to the same port used in your application
    publisher.bind("tcp://*:5556")
    
    # Create a synchronization socket
    sync_socket = context.socket(zmq.REP)
    sync_socket.bind("tcp://*:5557")
    
    print("Publisher: Waiting for subscribers to connect...")
    
    # Wait for synchronization request (with timeout)
    poller = zmq.Poller()
    poller.register(sync_socket, zmq.POLLIN)
    
    if poller.poll(5000):  # 5 second timeout
        sync_msg = sync_socket.recv_string()
        print(f"Publisher: Received sync request: {sync_msg}")
        sync_socket.send_string("READY")
        print("Publisher: Sent READY response")
    else:
        print("Publisher: No subscribers connected within timeout period")
    
    # Start publishing messages
    count = 0
    try:
        while True:
            message = {"count": count, "timestamp": time.time()}
            publisher.send_json(message)
            print(f"Publisher: Sent message {count}")
            count += 1
            time.sleep(1)
    except KeyboardInterrupt:
        print("Publisher: Stopping...")
    finally:
        publisher.close()
        sync_socket.close()
        context.term()

def run_subscriber():
    """Run a simple ZMQ subscriber"""
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    
    # Connect to the publisher
    subscriber.connect("tcp://localhost:5556")
    subscriber.subscribe("")  # Subscribe to all messages
    
    # Create synchronization socket
    sync_socket = context.socket(zmq.REQ)
    sync_socket.connect("tcp://localhost:5557")
    
    # Send synchronization request
    print("Subscriber: Sending sync request...")
    sync_socket.send_string("SYNC")
    
    # Wait for response with timeout
    poller = zmq.Poller()
    poller.register(sync_socket, zmq.POLLIN)
    
    if poller.poll(5000):  # 5 second timeout
        response = sync_socket.recv_string()
        print(f"Subscriber: Received sync response: {response}")
    else:
        print("Subscriber: Sync timeout - no response from publisher")
        return
    
    # Close sync socket
    sync_socket.close()
    
    # Set up polling for subscriber
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    
    # Receive messages
    try:
        print("Subscriber: Waiting for messages...")
        while True:
            socks = dict(poller.poll(1000))  # 1 second timeout
            if subscriber in socks and socks[subscriber] == zmq.POLLIN:
                message = subscriber.recv_json()
                print(f"Subscriber: Received message: {message}")
            else:
                print("Subscriber: No message received (timeout)")
    except KeyboardInterrupt:
        print("Subscriber: Stopping...")
    finally:
        subscriber.close()
        context.term()

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python test_zmq_connection.py [pub|sub]")
        return 1
    
    mode = sys.argv[1].lower()
    
    if mode == "pub":
        run_publisher()
    elif mode == "sub":
        run_subscriber()
    else:
        print(f"Unknown mode: {mode}. Use 'pub' or 'sub'")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())