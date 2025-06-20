#!/usr/bin/env python3
"""
CurrentRamp ADC Streaming Test
Tests the ADC streaming functionality added to the CurrentRamp driver
Goal: Capture 10+ seconds of ADC data while voltage ramps are running
"""

import os
import time
import sys
import socket
import struct
import threading
import queue
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver - ADC streaming functions only"""
    
    def __init__(self, client):
        self.client = client

    # ADC Streaming Functions (the ones I just added)
    @command()
    def start_streaming(self, host_ip, port):
        """Start ADC streaming to specified host and port"""
        pass
    
    @command()
    def stop_streaming(self):
        """Stop ADC streaming"""
        pass
    
    @command()
    def get_streaming_enabled(self):
        """Get streaming enable status"""
        return self.client.recv_bool()
    
    @command()
    def get_stream_packets_sent(self):
        """Get number of packets sent"""
        return self.client.recv_uint64()
    
    @command()
    def get_stream_samples_sent(self):
        """Get number of samples sent"""
        return self.client.recv_uint64()

class UdpReceiver:
    """UDP receiver for ADC streaming data"""
    
    def __init__(self, port=12345):
        self.port = port
        self.socket = None
        self.running = False
        self.data_queue = queue.Queue()
        self.stats = {
            'packets_received': 0,
            'samples_received': 0,
            'last_sequence': -1
        }
        
    def start(self):
        """Start UDP receiver"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.settimeout(1.0)
            self.running = True
            
            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()
            print(f"   📡 UDP receiver started on port {self.port}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to start UDP receiver: {e}")
            return False
    
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
                data, addr = self.socket.recvfrom(8192)
                packet = self._parse_packet(data)
                if packet:
                    self.data_queue.put(packet, block=False)
            except socket.timeout:
                continue
            except queue.Full:
                pass  # Drop packets if queue is full
            except Exception as e:
                if self.running:
                    print(f"   ⚠️ Receive error: {e}")
                break
    
    def _parse_packet(self, data):
        """Parse UDP packet with header + samples"""
        if len(data) < 16:  # Minimum header size
            return None
            
        try:
            # Parse header: sequence (8 bytes) + timestamp (8 bytes) as uint16_t values
            header = struct.unpack('<8H', data[:16])
            
            # Reconstruct 64-bit values
            sequence = (header[3] << 48) | (header[2] << 32) | (header[1] << 16) | header[0]
            timestamp = (header[7] << 48) | (header[6] << 32) | (header[5] << 16) | header[4]
            
            # Check for dropped packets
            if self.stats['last_sequence'] >= 0:
                expected = self.stats['last_sequence'] + 1
                if sequence != expected and sequence > expected:
                    print(f"   ⚠️ Dropped {sequence - expected} packets")
            
            self.stats['last_sequence'] = sequence
            
            # Parse samples (16-bit values)
            sample_data = data[16:]
            if len(sample_data) % 2 != 0:
                return None
                
            samples = struct.unpack(f'<{len(sample_data)//2}H', sample_data)
            
            # Convert to voltage (assuming ±1.8V range, 16-bit ADC)
            samples_voltage = [(s - 32768) * 1.8 / 32768.0 if s >= 32768 else s * 1.8 / 32768.0 for s in samples]
            
            self.stats['packets_received'] += 1
            self.stats['samples_received'] += len(samples)
            
            return {
                'sequence': sequence,
                'timestamp': timestamp,
                'samples': np.array(samples_voltage)
            }
            
        except struct.error:
            return None
    
    def get_stats(self):
        """Get receiver statistics"""
        return self.stats.copy()

def connect_to_device():
    """Connect to Alpha250 currentramp instrument"""
    print("🔌 Connecting to Alpha250 currentramp...")
    
    try:
        host = os.environ.get('HOST', '192.168.1.100')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"   ✅ Connected to {host}")
        return driver
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None

def test_streaming_basic(driver):
    """Test basic streaming start/stop"""
    print("\n🧪 Testing Basic ADC Streaming...")
    
    try:
        # Check initial state
        enabled = driver.get_streaming_enabled()
        print(f"   Initial streaming state: {enabled}")
        
        if enabled:
            print("   Stopping existing stream...")
            driver.stop_streaming()
            time.sleep(0.5)
        
        # Start streaming
        print("   Starting streaming to localhost:12345...")
        driver.start_streaming("127.0.0.1", 12345)
        time.sleep(0.5)
        
        # Check if started
        enabled = driver.get_streaming_enabled()
        print(f"   Streaming enabled: {enabled}")
        
        if enabled:
            # Let it run for a few seconds
            print("   Streaming for 3 seconds...")
            time.sleep(3.0)
            
            # Check stats
            packets = driver.get_stream_packets_sent()
            samples = driver.get_stream_samples_sent()
            print(f"   Packets sent: {packets}")
            print(f"   Samples sent: {samples}")
            
            # Stop streaming
            print("   Stopping streaming...")
            driver.stop_streaming()
            time.sleep(0.5)
            
            enabled = driver.get_streaming_enabled()
            print(f"   Streaming enabled after stop: {enabled}")
            
            if not enabled:
                print("   ✅ Basic streaming control working")
                return True
            else:
                print("   ❌ Failed to stop streaming")
                return False
        else:
            print("   ❌ Failed to start streaming")
            return False
            
    except Exception as e:
        print(f"   ❌ Basic streaming test failed: {e}")
        return False

def test_streaming_data_capture(driver):
    """Test actual data capture for 10+ seconds"""
    print("\n🧪 Testing 10+ Second Data Capture...")
    
    try:
        # Setup UDP receiver
        receiver = UdpReceiver(port=12346)
        if not receiver.start():
            return False
        
        # Start streaming
        print("   Starting 15-second data capture...")
        driver.start_streaming("127.0.0.1", 12346)
        time.sleep(0.5)
        
        # Check streaming started
        enabled = driver.get_streaming_enabled()
        if not enabled:
            print("   ❌ Failed to start streaming")
            receiver.stop()
            return False
        
        # Collect data for 15 seconds
        all_samples = []
        start_time = time.time()
        last_stats_time = start_time
        
        print("   Collecting data...")
        while time.time() - start_time < 15.0:
            # Process received packets
            packets_processed = 0
            while not receiver.data_queue.empty() and packets_processed < 10:
                try:
                    packet = receiver.data_queue.get_nowait()
                    all_samples.extend(packet['samples'])
                    packets_processed += 1
                except queue.Empty:
                    break
            
            # Print stats every 2 seconds
            current_time = time.time()
            if current_time - last_stats_time >= 2.0:
                elapsed = current_time - start_time
                device_packets = driver.get_stream_packets_sent()
                device_samples = driver.get_stream_samples_sent()
                receiver_stats = receiver.get_stats()
                
                print(f"     {elapsed:.1f}s: Device sent {device_packets} packets, {device_samples} samples")
                print(f"           Received {receiver_stats['packets_received']} packets, {len(all_samples)} samples")
                
                last_stats_time = current_time
            
            time.sleep(0.1)
        
        # Stop streaming
        print("   Stopping streaming...")
        driver.stop_streaming()
        receiver.stop()
        time.sleep(0.5)
        
        # Final results
        total_time = time.time() - start_time
        print(f"\n   📊 Capture Results:")
        print(f"     Duration: {total_time:.1f} seconds")
        print(f"     Total samples: {len(all_samples):,}")
        print(f"     Sample rate: {len(all_samples)/total_time:.1f} Hz")
        
        if len(all_samples) > 0:
            data = np.array(all_samples)
            print(f"     Voltage range: {np.min(data):.3f} to {np.max(data):.3f} V")
            print(f"     Mean voltage: {np.mean(data):.3f} V")
            print(f"     Std deviation: {np.std(data):.3f} V")
            
            # Check if we got reasonable amount of data
            expected_samples = 10000 * total_time  # 10 kHz target rate
            if len(all_samples) > expected_samples * 0.8:  # At least 80% of expected
                print("   ✅ Long-duration capture working")
                return True
            else:
                print("   ⚠️ Lower sample rate than expected")
                return False
        else:
            print("   ❌ No data received")
            return False
            
    except Exception as e:
        print(f"   ❌ Data capture test failed: {e}")
        return False

def test_streaming_with_plotting(driver):
    """Test streaming with real-time plotting capability"""
    print("\n🧪 Testing Streaming with Data Visualization...")
    
    try:
        # Setup UDP receiver
        receiver = UdpReceiver(port=12347)
        if not receiver.start():
            return False
        
        # Start streaming
        print("   Starting streaming for plotting test...")
        driver.start_streaming("127.0.0.1", 12347)
        time.sleep(0.5)
        
        # Collect data for 5 seconds
        all_samples = []
        timestamps = []
        start_time = time.time()
        
        while time.time() - start_time < 5.0:
            # Process packets
            while not receiver.data_queue.empty():
                try:
                    packet = receiver.data_queue.get_nowait()
                    # Create timestamps for samples (assuming 10 kHz rate)
                    packet_start_time = len(all_samples) / 10000.0
                    for i, sample in enumerate(packet['samples']):
                        all_samples.append(sample)
                        timestamps.append(packet_start_time + i / 10000.0)
                except queue.Empty:
                    break
            
            time.sleep(0.1)
        
        # Stop streaming
        driver.stop_streaming()
        receiver.stop()
        
        if len(all_samples) > 1000:
            print(f"   📊 Collected {len(all_samples)} samples for plotting")
            
            # Create a simple plot
            plt.figure(figsize=(12, 6))
            
            # Plot first 1000 points for clarity
            n_plot = min(1000, len(all_samples))
            plt.plot(timestamps[:n_plot], all_samples[:n_plot], 'b-', linewidth=0.8)
            plt.xlabel('Time (seconds)')
            plt.ylabel('ADC Voltage (V)')
            plt.title('CurrentRamp ADC Streaming Data')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = "adc_streaming_test.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"   💾 Plot saved as {plot_file}")
            print("   ✅ Plotting capability working")
            return True
        else:
            print("   ❌ Insufficient data for plotting")
            return False
            
    except Exception as e:
        print(f"   ❌ Plotting test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 CurrentRamp ADC Streaming Test Suite")
    print("   Testing 10+ second ADC data capture capability")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("\n❌ Cannot connect to currentramp instrument. Exiting.")
        print("💡 Make sure the currentramp instrument is built and running")
        return 1
    
    # Run streaming tests
    tests = [
        ("Basic Streaming", test_streaming_basic),
        ("10+ Second Capture", test_streaming_data_capture),
        ("Data Visualization", test_streaming_with_plotting),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(driver)
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ADC Streaming Test Results:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ADC streaming is working!")
        print("📊 You can now capture 10+ seconds of ADC data while your ramps run")
        print("💡 Next: Use the stream_monitor.py for real-time visualization")
        return 0
    else:
        print("⚠️ Some streaming tests failed.")
        print("🔧 Check that the currentramp instrument includes the streaming code")
        return 1

if __name__ == "__main__":
    exit(main()) 