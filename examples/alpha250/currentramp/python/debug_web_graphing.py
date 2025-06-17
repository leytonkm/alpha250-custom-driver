#!/usr/bin/env python3
"""
Debug Web Graphing for CurrentRamp
Quick test to identify issues with the voltage vs voltage graphing
"""

import os
import time
import numpy as np
from koheron import connect, command

class CurrentRamp:
    """CurrentRamp Python interface for debugging"""
    
    def __init__(self, client):
        self.client = client

    @command()
    def get_photodiode_precision(self, channel):
        """Get precision ADC reading from specified channel (0-7)"""
        return self.client.recv_float()

    @command()
    def get_ramp_offset(self):
        """Get current ramp offset"""
        return self.client.recv_float()

    @command()
    def get_ramp_amplitude(self):
        """Get current ramp amplitude"""
        return self.client.recv_float()

    @command()
    def get_ramp_frequency(self):
        """Get current ramp frequency"""
        return self.client.recv_double()

    @command()
    def get_ramp_enabled(self):
        """Check if ramp is enabled"""
        return self.client.recv_bool()

    @command()
    def start_ramp(self):
        """Start the ramp"""
        pass

    @command()
    def generate_ramp_waveform(self):
        """Generate ramp waveform"""
        pass

def debug_web_graphing():
    """Debug the web graphing functionality"""
    print("🔧 Web Graphing Debug")
    print("=" * 40)
    
    # Connect to device
    try:
        host = os.getenv('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"✅ Connected to {host}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    # Test 1: Check precision ADC readings
    print(f"\n📊 Test 1: Precision ADC Readings")
    try:
        for channel in range(4):
            voltage = driver.get_photodiode_precision(channel)
            print(f"  Channel {channel}: {voltage:.4f}V")
    except Exception as e:
        print(f"❌ ADC reading failed: {e}")
        return

    # Test 2: Check ramp parameters
    print(f"\n⚙️  Test 2: Ramp Parameters")
    try:
        offset = driver.get_ramp_offset()
        amplitude = driver.get_ramp_amplitude()
        frequency = driver.get_ramp_frequency()
        enabled = driver.get_ramp_enabled()
        
        print(f"  Offset: {offset:.3f}V")
        print(f"  Amplitude: {amplitude:.3f}V")
        print(f"  Frequency: {frequency:.1f}Hz")
        print(f"  Enabled: {enabled}")
        
        if not enabled:
            print("⚠️  Ramp is not enabled - this could be why graph is empty")
            print("   Try enabling the ramp in the web interface first")
    except Exception as e:
        print(f"❌ Ramp parameter check failed: {e}")
        return

    # Test 3: Simulate web graphing logic
    print(f"\n🎯 Test 3: Simulating Web Graphing Logic")
    try:
        # Get current parameters
        offset = driver.get_ramp_offset()
        amplitude = driver.get_ramp_amplitude()
        frequency = driver.get_ramp_frequency()
        enabled = driver.get_ramp_enabled()
        
        print(f"  Current time: {time.time():.3f}")
        
        # Calculate DAC voltage (same logic as web interface)
        if enabled:
            current_time = time.time()
            ramp_phase = (current_time * frequency) % 1.0
            dac_voltage = offset + (amplitude * ramp_phase)
            print(f"  Calculated DAC voltage: {dac_voltage:.4f}V")
            print(f"  Ramp phase: {ramp_phase:.4f}")
        else:
            dac_voltage = offset
            print(f"  DAC voltage (ramp disabled): {dac_voltage:.4f}V")
        
        # Get ADC reading
        adc_voltage = driver.get_photodiode_precision(0)  # Channel 0
        print(f"  ADC voltage (channel 0): {adc_voltage:.4f}V")
        
        # Check if values are reasonable
        if abs(adc_voltage) < 0.001:
            print("⚠️  ADC reading is very close to zero - check connections")
        
        if not enabled:
            print("⚠️  Ramp is disabled - DAC voltage won't change over time")
            
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return

    # Test 4: Check web interface elements
    print(f"\n🌐 Test 4: Web Interface Checklist")
    print("  Check these in your browser:")
    print("  1. Is the 'Live Voltage vs Voltage Analysis' section visible?")
    print("  2. Is the plot placeholder div present?")
    print("  3. Are there any JavaScript console errors?")
    print("  4. Is the 'Enable Live Graphing' toggle working?")
    print("  5. Do the sampling rate and data points controls work?")
    
    # Test 5: Rapid sampling test
    print(f"\n⚡ Test 5: Rapid Sampling Test (10 samples)")
    try:
        samples = []
        for i in range(10):
            current_time = time.time()
            
            # Get fresh parameters each time
            offset = driver.get_ramp_offset()
            amplitude = driver.get_ramp_amplitude()
            frequency = driver.get_ramp_frequency()
            enabled = driver.get_ramp_enabled()
            
            # Calculate DAC
            if enabled:
                ramp_phase = (current_time * frequency) % 1.0
                dac_voltage = offset + (amplitude * ramp_phase)
            else:
                dac_voltage = offset
            
            # Get ADC
            adc_voltage = driver.get_photodiode_precision(0)
            
            samples.append((dac_voltage, adc_voltage))
            print(f"  Sample {i+1}: DAC={dac_voltage:.4f}V, ADC={adc_voltage:.4f}V")
            
            time.sleep(0.1)  # 100ms between samples
        
        # Analyze samples
        dac_values = [s[0] for s in samples]
        adc_values = [s[1] for s in samples]
        
        dac_range = max(dac_values) - min(dac_values)
        adc_range = max(adc_values) - min(adc_values)
        
        print(f"\n📈 Sample Analysis:")
        print(f"  DAC range: {dac_range:.4f}V")
        print(f"  ADC range: {adc_range:.4f}V")
        
        if dac_range < 0.001:
            print("⚠️  DAC values not changing - ramp may not be working")
        if adc_range < 0.001:
            print("⚠️  ADC values not changing - check input signal")
            
    except Exception as e:
        print(f"❌ Rapid sampling test failed: {e}")

    print(f"\n✅ Debug complete!")
    print(f"\n💡 Common Issues:")
    print(f"  - Ramp not enabled: Enable ramp in web interface")
    print(f"  - No input signal: Check sine wave connection to ADC channel 0")
    print(f"  - JavaScript errors: Check browser console (F12)")
    print(f"  - Plot not initializing: Check if plot placeholder div exists")
    print(f"  - Timing issues: Try different sampling rates")

if __name__ == '__main__':
    debug_web_graphing() 