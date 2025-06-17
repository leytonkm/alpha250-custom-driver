#!/usr/bin/env python3

import numpy as np
import time
import sys
import os

# Add the parent directory to the path so we can import the driver
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from currentramp import CurrentRamp

def debug_time_series():
    """Debug time-series data collection similar to web interface"""
    
    # Connect to the device
    host = os.environ.get('HOST', '192.168.1.20')
    currentramp = CurrentRamp(host)
    
    print("=== Time-Series Debug ===")
    print(f"Connected to {host}")
    
    # Check current ramp settings
    print("\n=== Current Ramp Settings ===")
    print(f"Ramp Enabled: {currentramp.get_ramp_enabled()}")
    print(f"Frequency: {currentramp.get_ramp_frequency():.1f} Hz")
    print(f"Amplitude: {currentramp.get_ramp_amplitude():.3f} V")
    print(f"Offset: {currentramp.get_ramp_offset():.3f} V")
    
    # Collect time-series data
    print("\n=== Collecting Time-Series Data ===")
    print("Collecting 50 samples over 1 second...")
    
    start_time = time.time()
    times = []
    dac_voltages = []
    adc_voltages = []
    
    for i in range(50):
        current_time = time.time() - start_time
        
        # Calculate expected DAC voltage (same logic as web interface)
        if currentramp.get_ramp_enabled():
            frequency = currentramp.get_ramp_frequency()
            amplitude = currentramp.get_ramp_amplitude()
            offset = currentramp.get_ramp_offset()
            
            ramp_phase = (current_time * frequency) % 1.0  # 0 to 1
            dac_voltage = offset + (amplitude * ramp_phase)
        else:
            dac_voltage = currentramp.get_ramp_offset()
        
        # Get ADC reading
        adc_voltage = currentramp.get_photodiode_precision(0)
        
        times.append(current_time)
        dac_voltages.append(dac_voltage)
        adc_voltages.append(adc_voltage)
        
        print(f"t={current_time:.3f}s: DAC={dac_voltage:.3f}V, ADC={adc_voltage:.3f}V")
        
        time.sleep(0.02)  # 20ms like web interface
    
    print("\n=== Data Summary ===")
    print(f"Time range: {min(times):.3f} to {max(times):.3f} seconds")
    print(f"DAC range: {min(dac_voltages):.3f} to {max(dac_voltages):.3f} V")
    print(f"ADC range: {min(adc_voltages):.3f} to {max(adc_voltages):.3f} V")
    
    # Check if data is varying
    dac_variation = max(dac_voltages) - min(dac_voltages)
    adc_variation = max(adc_voltages) - min(adc_voltages)
    
    print(f"\n=== Variation Analysis ===")
    print(f"DAC variation: {dac_variation:.3f} V")
    print(f"ADC variation: {adc_variation:.3f} V")
    
    if dac_variation < 0.01:
        print("⚠️  WARNING: DAC shows little variation - ramp may not be working")
    else:
        print("✅ DAC shows good variation - ramp is working")
        
    if adc_variation < 0.01:
        print("⚠️  WARNING: ADC shows little variation - input signal may be missing")
    else:
        print("✅ ADC shows good variation - input signal detected")
    
    print("\n=== Web Interface Debugging Tips ===")
    print("1. Check browser console for JavaScript errors")
    print("2. Verify ramp is enabled before starting graphing")
    print("3. Try different time windows (2s, 10s, 30s)")
    print("4. Use 'Full History' mode to see all data")
    print("5. Check that both DAC and ADC signals are varying")

if __name__ == "__main__":
    try:
        debug_time_series()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the device is connected and ramp is enabled") 