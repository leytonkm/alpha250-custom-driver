#!/usr/bin/env python3
"""
Simple ADC Test for CurrentRamp
Tests basic ADC reading using the simple functions I just added
Goal: Read ADC data to verify the ADC is working
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver - Basic ADC functions"""
    
    def __init__(self, client):
        self.client = client

    # Basic ADC reading functions
    @command()
    def get_adc0(self):
        """Get ADC0 raw value"""
        return self.client.recv_uint32()
    
    @command()
    def get_adc1(self):
        """Get ADC1 raw value"""
        return self.client.recv_uint32()
    
    @command()
    def get_adc0_voltage(self):
        """Get ADC0 voltage"""
        return self.client.recv_float()
    
    @command()
    def get_adc1_voltage(self):
        """Get ADC1 voltage"""
        return self.client.recv_float()

def connect_to_device():
    """Connect to Alpha250 currentramp instrument"""
    print("🔌 Connecting to Alpha250 currentramp...")
    
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"   ✅ Connected to {host}")
        return driver
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None

def test_basic_adc_reading(driver):
    """Test basic ADC reading"""
    print("\n🧪 Testing Basic ADC Reading...")
    
    try:
        # Read ADC values multiple times
        print("   Reading ADC values 10 times...")
        
        adc0_values = []
        adc1_values = []
        adc0_voltages = []
        adc1_voltages = []
        
        for i in range(10):
            # Get raw values
            adc0_raw = driver.get_adc0()
            adc1_raw = driver.get_adc1()
            
            # Get voltage values
            adc0_volt = driver.get_adc0_voltage()
            adc1_volt = driver.get_adc1_voltage()
            
            adc0_values.append(adc0_raw)
            adc1_values.append(adc1_raw)
            adc0_voltages.append(adc0_volt)
            adc1_voltages.append(adc1_volt)
            
            print(f"     #{i+1}: ADC0={adc0_raw:5d} ({adc0_volt:+.3f}V), ADC1={adc1_raw:5d} ({adc1_volt:+.3f}V)")
            
            time.sleep(0.1)
        
        # Statistics
        print(f"\n   📊 ADC0 Statistics:")
        print(f"     Raw range: {min(adc0_values)} to {max(adc0_values)}")
        print(f"     Voltage range: {min(adc0_voltages):+.3f}V to {max(adc0_voltages):+.3f}V")
        print(f"     Average: {np.mean(adc0_voltages):+.3f}V")
        
        print(f"\n   📊 ADC1 Statistics:")
        print(f"     Raw range: {min(adc1_values)} to {max(adc1_values)}")
        print(f"     Voltage range: {min(adc1_voltages):+.3f}V to {max(adc1_voltages):+.3f}V")
        print(f"     Average: {np.mean(adc1_voltages):+.3f}V")
        
        print("   ✅ Basic ADC reading working")
        return True, (adc0_voltages, adc1_voltages)
        
    except Exception as e:
        print(f"   ❌ ADC reading test failed: {e}")
        return False, None

def test_continuous_reading(driver, duration=10):
    """Test continuous ADC reading for specified duration"""
    print(f"\n🧪 Testing Continuous ADC Reading for {duration} seconds...")
    
    try:
        start_time = time.time()
        samples = []
        timestamps = []
        
        print("   Collecting data...")
        sample_count = 0
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            # Read ADC0 (you can change to ADC1 if needed)
            voltage = driver.get_adc0_voltage()
            
            samples.append(voltage)
            timestamps.append(current_time)
            sample_count += 1
            
            # Print progress every 100 samples
            if sample_count % 100 == 0:
                print(f"     {current_time:.1f}s: {sample_count} samples, latest = {voltage:+.3f}V")
            
            time.sleep(0.01)  # 100 Hz sampling rate
        
        total_time = time.time() - start_time
        sample_rate = len(samples) / total_time
        
        print(f"\n   📊 Continuous Reading Results:")
        print(f"     Duration: {total_time:.1f} seconds")
        print(f"     Total samples: {len(samples)}")
        print(f"     Sample rate: {sample_rate:.1f} Hz")
        print(f"     Voltage range: {min(samples):+.3f}V to {max(samples):+.3f}V")
        print(f"     Average voltage: {np.mean(samples):+.3f}V")
        print(f"     Std deviation: {np.std(samples):.3f}V")
        
        print("   ✅ Continuous reading working")
        return True, (timestamps, samples)
        
    except Exception as e:
        print(f"   ❌ Continuous reading test failed: {e}")
        return False, None

def plot_adc_data(timestamps, samples, title="ADC Data"):
    """Plot ADC data"""
    if not timestamps or not samples:
        print("   No data to plot")
        return
    
    print(f"   📊 Plotting {len(samples)} samples...")
    
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, samples, 'b-', linewidth=0.8)
    plt.xlabel('Time (seconds)')
    plt.ylabel('ADC Voltage (V)')
    plt.title(f'CurrentRamp {title}')
    plt.grid(True, alpha=0.3)
    
    # Add statistics to plot
    plt.text(0.02, 0.98, 
             f'Samples: {len(samples)}\nRange: {min(samples):+.3f}V to {max(samples):+.3f}V\nMean: {np.mean(samples):+.3f}V',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save plot
    plot_file = f"currentramp_{title.lower().replace(' ', '_')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Plot saved as {plot_file}")

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 CurrentRamp Simple ADC Test")
    print("   Testing basic ADC reading to verify functionality")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("\n❌ Cannot connect to currentramp instrument. Exiting.")
        return 1
    
    # Test basic ADC reading
    success, data = test_basic_adc_reading(driver)
    if not success:
        print("\n❌ Basic ADC reading failed. Check your setup.")
        return 1
    
    # Test continuous reading
    success, continuous_data = test_continuous_reading(driver, duration=10)
    if success and continuous_data:
        timestamps, samples = continuous_data
        plot_adc_data(timestamps, samples, "Continuous ADC Reading")
    
    print("\n" + "=" * 60)
    print("📊 Simple ADC Test Complete")
    print("✅ Your currentramp instrument can read ADC data!")
    print("🎯 This proves the basic ADC functionality is working")
    print("💡 Next step: Implement proper streaming for 10+ second capture")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main()) 