#!/usr/bin/env python3
"""
CurrentRamp ADC Stream Monitor
Real-time monitoring of ADC data during voltage ramp experiments
Receives UDP streamed data and provides plotting capabilities
"""

import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import time
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import signal
import sys

@dataclass
class StreamConfig:
    """Configuration for UDP streaming"""
    host: str = "0.0.0.0"
    port: int = 12345
    buffer_size: int = 8192
    max_plot_duration: float = 30.0  # seconds
    update_interval: int = 100  # milliseconds

class UDPReceiver:
    """UDP packet receiver with sequence tracking"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.socket = None
        self.running = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.stats = {
            'packets_received': 0,
            'samples_received': 0,
            'packets_dropped': 0,
            'last_sequence': -1
        }
        
    def start(self):
        """Start UDP receiver"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 1MB buffer
            self.socket.bind((self.config.host, self.config.port))
            self.socket.settimeout(1.0)  # 1 second timeout
            self.running = True
            
            print(f"🎧 Listening for UDP packets on {self.config.host}:{self.config.port}")
            
            # Start receiver thread
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            
        except Exception as e:
            print(f"❌ Failed to start UDP receiver: {e}")
            return False
        
        return True
    
    def stop(self):
        """Stop UDP receiver"""
        self.running = False
        if self.socket:
            self.socket.close()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2.0)
    
    def _receive_loop(self):
        """Main receive loop"""
        while self.running:
            try:
                # Receive packet
                data, addr = self.socket.recvfrom(self.config.buffer_size)
                
                # Parse packet
                packet = self._parse_packet(data)
                if packet:
                    self.data_queue.put(packet, block=False)
                    
            except socket.timeout:
                continue  # Normal timeout, keep looping
            except queue.Full:
                self.stats['packets_dropped'] += 1
                print("⚠️  Data queue full, dropping packets")
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    print(f"❌ Receive error: {e}")
                break
    
    def _parse_packet(self, data: bytes) -> Optional[dict]:
        """Parse UDP packet with header + samples"""
        if len(data) < 16:  # Minimum header size
            return None
            
        try:
            # Parse header: sequence (8 bytes) + timestamp (8 bytes)
            sequence, timestamp = struct.unpack('<QQ', data[:16])
            
            # Check for dropped packets
            if self.stats['last_sequence'] >= 0:
                expected = self.stats['last_sequence'] + 1
                if sequence != expected:
                    dropped = sequence - expected
                    self.stats['packets_dropped'] += dropped
                    if dropped > 0:
                        print(f"⚠️  Dropped {dropped} packets (expected {expected}, got {sequence})")
            
            self.stats['last_sequence'] = sequence
            
            # Parse samples (16-bit values)
            sample_data = data[16:]
            if len(sample_data) % 2 != 0:
                return None
                
            samples = struct.unpack(f'<{len(sample_data)//2}H', sample_data)
            
            # Convert to voltage (Alpha250: 14-bit ADC, ±500mV range, zero-padded to 16-bit)
            samples_14bit = [s & 0x3FFF for s in samples]  # Extract lower 14 bits
            samples_voltage = [((s / 16383.0) - 0.5) * 1.0 for s in samples_14bit]
            
            self.stats['packets_received'] += 1
            self.stats['samples_received'] += len(samples)
            
            return {
                'sequence': sequence,
                'timestamp': timestamp,
                'samples': np.array(samples_voltage),
                'raw_samples': np.array(samples)
            }
            
        except struct.error as e:
            print(f"❌ Packet parse error: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get receiver statistics"""
        return self.stats.copy()

if __name__ == "__main__":
    print("✅ Stream monitor created successfully!")
    print("📋 To run: python3 stream_monitor.py --duration 60 --port 12345")

class RealTimePlotter:
    """Real-time data plotter with configurable time window"""
    
    def __init__(self, config: StreamConfig, receiver: UDPReceiver):
        self.config = config
        self.receiver = receiver
        
        # Calculate sample rate (100 kHz fixed)
        self.sample_rate = 100000.0  # Hz
        self.max_samples = int(self.config.max_plot_duration * self.sample_rate)
        
        # Data buffers
        self.times = deque(maxlen=self.max_samples)
        self.voltages = deque(maxlen=self.max_samples)
        self.current_time = 0.0
        
        # Setup plot
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('CurrentRamp ADC Stream Monitor', fontsize=14, fontweight='bold')
        
        # Voltage plot
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=0.8)
        self.ax1.set_ylabel('ADC Voltage (V)')
        self.ax1.set_title('Real-time ADC Data')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_ylim(-2.0, 2.0)
        
        # Statistics plot  
        self.ax2.axis('off')
        self.stats_text = self.ax2.text(0.1, 0.5, '', fontsize=10, fontfamily='monospace')
        
        # Animation
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plot, 
            interval=self.config.update_interval,
            blit=False, cache_frame_data=False
        )
        
        # Statistics tracking
        self.start_time = time.time()
        self.last_stats_time = time.time()
        self.samples_per_second = 0.0
        
    def _update_plot(self, frame):
        """Update plot with new data"""
        # Process all available packets
        packets_processed = 0
        while not self.receiver.data_queue.empty() and packets_processed < 10:
            try:
                packet = self.receiver.data_queue.get_nowait()
                self._add_packet_data(packet)
                packets_processed += 1
            except queue.Empty:
                break
        
        # Update voltage plot
        if self.times and self.voltages:
            self.line1.set_data(list(self.times), list(self.voltages))
            
            # Auto-scale time axis
            if len(self.times) > 1:
                time_span = self.times[-1] - self.times[0]
                if time_span > 0:
                    margin = time_span * 0.05
                    self.ax1.set_xlim(self.times[0] - margin, self.times[-1] + margin)
        
        # Update statistics
        self._update_statistics()
        
        return [self.line1, self.stats_text]
    
    def _add_packet_data(self, packet: dict):
        """Add packet data to buffers"""
        samples = packet['samples']
        
        # Calculate time values for this packet
        dt = 1.0 / self.sample_rate
        packet_times = self.current_time + np.arange(len(samples)) * dt
        
        # Add to buffers
        self.times.extend(packet_times)
        self.voltages.extend(samples)
        
        # Update current time
        self.current_time = packet_times[-1] + dt
    
    def _update_statistics(self):
        """Update statistics display"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Calculate sample rate
        if current_time - self.last_stats_time > 1.0:
            stats = self.receiver.get_stats()
            self.samples_per_second = stats['samples_received'] / elapsed if elapsed > 0 else 0
            self.last_stats_time = current_time
        
        # Get current stats
        stats = self.receiver.get_stats()
        
        # Format statistics
        stats_text = f"""
📊 STREAMING STATISTICS
├─ Runtime:          {elapsed:.1f} seconds
├─ Packets received: {stats['packets_received']:,}
├─ Samples received: {stats['samples_received']:,}
├─ Packets dropped:  {stats['packets_dropped']:,}
├─ Sample rate:      {self.samples_per_second:.1f} Hz
├─ Data points:      {len(self.voltages):,}
└─ Time span:        {self.current_time:.1f} seconds

🎛️  CURRENT VALUES"""
        
        if self.voltages:
            stats_text += f"""
├─ Latest voltage:   {self.voltages[-1]:.3f} V
├─ Min voltage:      {min(self.voltages):.3f} V
├─ Max voltage:      {max(self.voltages):.3f} V
└─ Avg voltage:      {np.mean(self.voltages):.3f} V"""
        else:
            stats_text += "\n└─ No data received yet"
        
        self.stats_text.set_text(stats_text)
    
    def show(self):
        """Show the plot"""
        plt.tight_layout()
        plt.show()

class CurrentRampMonitor:
    """Main application class"""
    
    def __init__(self, config: StreamConfig):
        self.config = config
        self.receiver = UDPReceiver(config)
        self.plotter = None
        self.running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start monitoring"""
        print("🚀 Starting CurrentRamp Stream Monitor")
        print(f"📡 UDP: {self.config.host}:{self.config.port}")
        print(f"⏱️  Plot duration: {self.config.max_plot_duration}s")
        print(f"🔄 Update interval: {self.config.update_interval}ms")
        print()
        
        # Start UDP receiver
        if not self.receiver.start():
            return False
        
        # Create plotter
        self.plotter = RealTimePlotter(self.config, self.receiver)
        self.running = True
        
        print("✅ Monitor started! Close plot window to stop.")
        
        # Show plot (blocking)
        try:
            self.plotter.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        
        return True
    
    def stop(self):
        """Stop monitoring"""
        if self.running:
            print("🛑 Stopping monitor...")
            self.running = False
            self.receiver.stop()
            if self.plotter:
                plt.close('all')
            print("✅ Monitor stopped")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='CurrentRamp ADC Stream Monitor - Real-time data visualization',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--host', default='0.0.0.0', 
                       help='Host IP address to bind to')
    parser.add_argument('--port', type=int, default=12345,
                       help='UDP port to listen on')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Maximum plot duration in seconds')
    parser.add_argument('--update-rate', type=int, default=100,
                       help='Plot update interval in milliseconds')
    
    args = parser.parse_args()
    
    # Create configuration
    config = StreamConfig(
        host=args.host,
        port=args.port,
        max_plot_duration=args.duration,
        update_interval=args.update_rate
    )
    
    # Create and start monitor
    monitor = CurrentRampMonitor(config)
    
    try:
        monitor.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

# Redefine main execution
if __name__ == "__main__":
    if len(sys.argv) > 1:
        exit(main())
    else:
        print("✅ Stream monitor created successfully!")
        print("📋 To run: python3 stream_monitor.py --duration 60 --port 12345")
