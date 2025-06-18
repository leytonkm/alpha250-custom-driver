#!/usr/bin/env python3

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect
from currentramp import CurrentRamp

def test_adc_oscilloscope():
    """Test ADC oscilloscope functionality with sine wave input"""
    
    # Connect to the device
    host = os.getenv('HOST', '192.168.1.20')  # Your Alpha250 IP
    client = connect(host, 'currentramp', restart=False)
    driver = CurrentRamp(client)
    
    print("=== Alpha250 ADC Oscilloscope Test ===")
    print(f"Connected to {host}")
    
    try:
        # Get oscilloscope parameters
        sampling_rate = driver.get_adc_sampling_rate()
        buffer_size = driver.get_adc_buffer_size()
        
        print(f"ADC sampling rate: {sampling_rate/1e6:.1f} MHz")
        print(f"Buffer size: {buffer_size} samples")
        
        # Calculate time axis
        time_step = 1.0 / sampling_rate
        time_axis = np.arange(buffer_size) * time_step * 1e6  # Time in microseconds
        max_time = time_axis[-1]
        
        print(f"Time per sample: {time_step*1e9:.2f} ns")
        print(f"Total capture time: {max_time:.2f} μs")
        
        # Test with sine wave input (user should connect -0.5 to 0.5V, 10Hz)
        print("\nReady to capture ADC data...")
        print("Make sure you have a sine wave connected to ADC Channel 0")
        print("Expected: -0.5V to +0.5V, 10Hz")
        
        # Capture ADC data
        adc_data = driver.get_adc_data_volts()
        
        print(f"Captured {len(adc_data)} samples")
        print(f"Voltage range: {np.min(adc_data):.3f}V to {np.max(adc_data):.3f}V")
        print(f"RMS voltage: {np.sqrt(np.mean(adc_data**2)):.3f}V")
        
        # Find peak-to-peak
        vpp = np.max(adc_data) - np.min(adc_data)
        print(f"Peak-to-peak voltage: {vpp:.3f}V")
        
        # Simple frequency estimation using zero crossings
        zero_crossings = np.where(np.diff(np.signbit(adc_data)))[0]
        if len(zero_crossings) > 1:
            # Estimate frequency from zero crossings
            avg_period_samples = 2 * np.mean(np.diff(zero_crossings))  # Factor of 2 for full cycle
            estimated_freq = sampling_rate / avg_period_samples
            print(f"Estimated frequency: {estimated_freq:.1f} Hz")
        
        # Plot the data
        plt.figure(figsize=(12, 6))
        
        # Plot full capture
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, adc_data, 'b-', linewidth=0.8)
        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V)')
        plt.title('ADC Oscilloscope - Full Capture')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.6, 0.6)
        
        # Plot zoomed view (first 1000 samples)
        plt.subplot(1, 2, 2)
        zoom_samples = min(1000, len(adc_data))
        plt.plot(time_axis[:zoom_samples], adc_data[:zoom_samples], 'r-', linewidth=1.2)
        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V)')
        plt.title(f'ADC Oscilloscope - Zoomed (first {zoom_samples} samples)')
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.6, 0.6)
        
        plt.tight_layout()
        plt.savefig('adc_oscilloscope_test.png', dpi=150)
        plt.show()
        
        print("\n✅ ADC Oscilloscope test completed!")
        print("Plot saved as 'adc_oscilloscope_test.png'")
        print(f"\nYou can now:")
        print(f"   - Open web interface at http://{host}")
        print("   - Use the oscilloscope interface to view real-time data")
        print("   - Adjust time range and trigger settings")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure the bitstream is loaded and the server is running")

if __name__ == "__main__":
    test_adc_oscilloscope() 