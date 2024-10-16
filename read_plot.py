import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for interactive plots

import matplotlib.pyplot as plt
import numpy as np
import socket
import struct
from datetime import datetime, timedelta

# Configuration
TCP_IP = "192.168.1.190"
TCP_PORT = 101
HEADER = b'\x00\xFF\x00'
PACKET_SIZE = 3 + 8 + 64  # 3 bytes header + 8 bytes timestamp + 64 bytes data

# Full scale for differential pressure sensor in Pa
full_scale_psi = 0.1445092
full_scale_pa = full_scale_psi * 6894.76

# Set up the plot
plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()
lines = [ax.plot([], [], label=f"CH{i+1}")[0] for i in range(32)]  # Create a line for each channel
ax.set_xlim(0, 100)
ax.set_ylim(-100, 100)
ax.legend(loc="upper right")

def connect_to_scanner():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))
    return sock

def read_packet(sock):
    # First, find the header
    while True:
        byte = sock.recv(1)
        if byte == b'\x00':
            byte += sock.recv(2)
            if byte == HEADER:
                break
    
    # Now read the rest of the packet
    packet = HEADER + sock.recv(PACKET_SIZE - 3)
    return packet

def parse_packet(packet):
    if packet[:3] != HEADER:
        raise ValueError("Invalid packet: incorrect header")
    
    # Parse timestamp (two 32-bit numbers)
    seconds, subseconds = struct.unpack('<II', packet[3:11])
    timestamp = datetime.fromtimestamp(seconds) + timedelta(microseconds=subseconds)
    
    # Parse 32 channels (16-bit integers)
    channels = struct.unpack('<32H', packet[11:])
    
    # Convert channels to differential pressure in Pa
    pressures = [(value - 32768) * (full_scale_pa / 32768) for value in channels]
    
    return timestamp, pressures

def main():
    sock = connect_to_scanner()
    print("Connected to nanoDAQ LTS-32. Streaming data...")

    xdata = list(range(100))  # Rolling window for the x-axis (last 100 points)
    ydata = [[] for _ in range(32)]  # One list for each channel

    try:
        while True:
            packet = read_packet(sock)
            try:
                timestamp, pressure_data = parse_packet(packet)
                print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}]")
                for i, pressure in enumerate(pressure_data, 1):
                    print(f"{pressure:.2f}", end=', ')
                print("\n--------------------")
                
                # Update plot data
                for i, pressure in enumerate(pressure_data):
                    ydata[i].append(pressure)
                    if len(ydata[i]) > 100:  # Keep only the last 100 points
                        ydata[i].pop(0)
                    lines[i].set_data(xdata[-len(ydata[i]):], ydata[i])
                
                ax.relim()  # Update axis limits
                ax.autoscale_view()  # Rescale the view to fit new data
                plt.pause(0.01)  # Pause to update the plot
                
            except ValueError as e:
                print(f"Error parsing packet: {e}")
                print(f"Raw packet (hex): {packet.hex()}")
    except KeyboardInterrupt:
        print("\nStopping data stream.")
    except ConnectionError as e:
        print(f"Error: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
