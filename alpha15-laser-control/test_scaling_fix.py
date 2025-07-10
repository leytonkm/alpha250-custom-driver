#!/usr/bin/env python3
"""
Test script to verify ADC voltage and timing scaling fixes.
This script tests with a known 10Hz, ±0.5V sine wave input.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver"""
    
    def __init__(self, client):
        self.client = client

    @command()
    def start_adc_streaming(self):
        pass
    
    @command()
    def stop_adc_streaming(self):
        pass
    
    @command()
    def set_decimation_rate(self, rate):
        pass
    
    @command()
    def get_decimated_sample_rate(self):
        return self.client.recv_double()
    
    @command()
    def get_adc_stream_voltages(self, num_samples):
        return self.client.recv_vector(dtype='float32')
    
    @command()
    def set_adc_input_range(self, range_sel):
        pass
    
    @command()
    def select_adc_channel(self, channel):
        pass

def test_scaling():
    """Test voltage and timing scaling with known input signal"""
    
    # Connect to instrument
    host = os.getenv('HOST', '192.168.1.115')
    print(f"Connecting to {host}...")
    
    client = connect(host, name='alpha15-laser-control')
    driver = CurrentRamp(client)
    
    # Configure ADC
    driver.select_adc_channel(0)  # Use channel 0
    driver.set_adc_input_range(0)  # Set to 2Vpp range for ±0.5V signal
    
    # Set decimation for good time resolution  
    decimation_rate = 100  # 15MHz / (100 × 2) = 75kS/s (factor of 2 from FIR)
    driver.set_decimation_rate(decimation_rate)
    
    # Get actual sample rate
    fs = driver.get_decimated_sample_rate()
    print(f"Decimated sample rate: {fs/1000:.1f} kS/s")
    
    # Calculate samples needed for 2 seconds of data
    duration_s = 2.0
    num_samples = int(fs * duration_s)
    print(f"Collecting {num_samples} samples over {duration_s:.1f} seconds...")
    
    # Start streaming and collect data
    driver.start_adc_streaming()
    time.sleep(0.5)  # Let DMA fill up
    
    data = np.array(driver.get_adc_stream_voltages(num_samples))
    driver.stop_adc_streaming()
    
    if len(data) == 0:
        print("❌ No data received!")
        return
        
    print(f"✅ Received {len(data)} samples")
    
    # Create time axis
    t = np.arange(len(data)) / fs
    
    # Analyze voltage scaling
    voltage_min = np.min(data)
    voltage_max = np.max(data)
    voltage_pp = voltage_max - voltage_min
    voltage_rms = np.std(data)
    
    print(f"\n📊 Voltage Analysis:")
    print(f"   Min: {voltage_min:.3f} V")
    print(f"   Max: {voltage_max:.3f} V") 
    print(f"   Peak-to-peak: {voltage_pp:.3f} V")
    print(f"   RMS: {voltage_rms:.3f} V")
    print(f"   Expected: ±0.5V (1.0V peak-to-peak)")
    
    # Check voltage scaling
    expected_pp = 1.0  # ±0.5V = 1V peak-to-peak
    voltage_error = abs(voltage_pp - expected_pp) / expected_pp * 100
    
    if voltage_error < 10:  # Within 10%
        print(f"✅ Voltage scaling: PASS (error: {voltage_error:.1f}%)")
    else:
        print(f"❌ Voltage scaling: FAIL (error: {voltage_error:.1f}%)")
    
    # Analyze timing by detecting zero crossings
    zero_crossings = []
    for i in range(1, len(data)):
        if (data[i-1] <= 0 < data[i]) or (data[i-1] >= 0 > data[i]):
            # Linear interpolation to find exact crossing
            t_cross = t[i-1] + (0 - data[i-1]) / (data[i] - data[i-1]) * (t[i] - t[i-1])
            zero_crossings.append(t_cross)
    
    if len(zero_crossings) >= 4:
        # Calculate period from zero crossings (4 crossings = 2 periods)
        periods = []
        for i in range(2, len(zero_crossings), 2):  # Every other crossing (same direction)
            period = zero_crossings[i] - zero_crossings[i-2]
            periods.append(period)
        
        avg_period = np.mean(periods)
        measured_freq = 1.0 / avg_period
        
        print(f"\n⏱️  Timing Analysis:")
        print(f"   Measured frequency: {measured_freq:.2f} Hz")
        print(f"   Expected frequency: 10.0 Hz")
        
        freq_error = abs(measured_freq - 10.0) / 10.0 * 100
        
        if freq_error < 5:  # Within 5%
            print(f"✅ Timing scaling: PASS (error: {freq_error:.1f}%)")
        else:
            print(f"❌ Timing scaling: FAIL (error: {freq_error:.1f}%)")
    else:
        print(f"❌ Timing analysis: Could not detect enough zero crossings")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot full signal
    plt.subplot(2, 1, 1)
    plt.plot(t, data, 'b-', linewidth=0.5)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Expected +0.5V')
    plt.axhline(y=-0.5, color='r', linestyle='--', alpha=0.7, label='Expected -0.5V')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(f'ADC Signal - Full Capture ({len(data)} samples @ {fs/1000:.1f} kS/s)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot zoomed view of first 0.5 seconds
    zoom_samples = int(0.5 * fs)
    plt.subplot(2, 1, 2)
    plt.plot(t[:zoom_samples], data[:zoom_samples], 'b-', linewidth=1)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=-0.5, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('ADC Signal - First 0.5 seconds (zoomed)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adc_scaling_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📈 Plot saved as 'adc_scaling_test.png'")

if __name__ == "__main__":
    test_scaling() 