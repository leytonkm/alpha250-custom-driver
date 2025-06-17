#!/usr/bin/env python3
"""
Simple ADC Input Test for CurrentRamp
Quick test to verify ADC functionality
"""

import os
import time
import numpy as np
from koheron import connect, command
import matplotlib.pyplot as plt

class CurrentRamp:
    """CurrentRamp Python interface for ADC testing"""
    
    def __init__(self, client):
        self.client = client

    @command()
    def get_photodiode_voltage(self):
        """Get photodiode voltage reading"""
        return self.client.recv_float()
    
    @command()
    def get_photodiode_raw(self):
        """Get raw photodiode ADC reading"""
        return self.client.recv_int16()
    
    @command()
    def set_photodiode_channel(self, channel):
        """Set photodiode ADC channel (0 or 1)"""
        pass
    
    @command()
    def get_photodiode_channel(self):
        """Get current photodiode ADC channel"""
        return self.client.recv_uint32()
    
    @command()
    def trigger_acquisition(self):
        """Trigger BRAM data acquisition"""
        pass
    
    @command()
    def get_adc_samples_1000(self):
        """Get 1000 ADC samples from BRAM"""
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def get_dac_samples_1000(self):
        """Get 1000 DAC samples from BRAM"""
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def set_ramp_frequency(self, frequency):
        """Set ramp frequency"""
        pass
    
    @command()
    def set_ramp_amplitude(self, amplitude):
        """Set ramp amplitude"""
        pass
    
    @command()
    def set_ramp_offset(self, offset):
        """Set ramp offset"""
        pass
    
    @command()
    def start_ramp(self):
        """Start ramp"""
        pass
    
    @command()
    def stop_ramp(self):
        """Stop ramp"""
        pass

    @command()
    def is_acquisition_complete(self):
        """Check if acquisition is complete"""
        return self.client.recv_bool()
    
    @command()
    def get_ramp_enabled(self):
        """Get ramp enable status"""
        return self.client.recv_bool()
    
    @command()
    def generate_ramp_waveform(self):
        """Configure ramp waveform parameters"""
        pass

def test_adc_readings(driver):
    """Test basic ADC readings"""
    print("📊 Testing ADC Readings...")
    
    # Make sure ramp is off for baseline readings
    print("  Ensuring ramp is stopped for baseline readings...")
    try:
        driver.stop_ramp()
        time.sleep(0.2)
    except:
        pass
    
    # Test both ADC channels
    for channel in [0, 1]:
        print(f"\n  Testing ADC Channel {channel}:")
        driver.set_photodiode_channel(channel)
        time.sleep(0.1)
        
        # Get several readings
        readings = []
        raw_readings = []
        for i in range(10):
            voltage = driver.get_photodiode_voltage()
            raw = driver.get_photodiode_raw()
            readings.append(voltage)
            raw_readings.append(raw)
            print(f"    Reading {i+1}: {voltage:.4f}V (raw: {raw})")
            time.sleep(0.05)
        
        # Check if readings are changing (not stuck)
        readings_array = np.array(readings)
        raw_array = np.array(raw_readings)
        voltage_range = np.max(readings_array) - np.min(readings_array)
        voltage_mean = np.mean(readings_array)
        raw_range = np.max(raw_array) - np.min(raw_array)
        raw_mean = np.mean(raw_array)
        
        print(f"    Voltage - Range: {voltage_range:.4f}V, Mean: {voltage_mean:.4f}V")
        print(f"    Raw     - Range: {raw_range} LSB, Mean: {raw_mean:.1f} LSB")
        
        # Analysis
        if voltage_range > 0.001:  # More than 1mV variation
            print(f"    ✅ Channel {channel}: ADC shows variation (likely working)")
        else:
            print(f"    ⚠️  Channel {channel}: ADC shows minimal variation")
            if abs(voltage_mean) < 0.01:
                print(f"    💡 Voltage near zero - no input signal (expected without ramp)")
            else:
                print(f"    💡 Stable voltage {voltage_mean:.4f}V - possibly a DC offset")
    
    return True

def test_bram_capture(driver):
    """Test BRAM data capture"""
    print("\n📈 Testing BRAM Data Capture...")
    
    try:
        # Trigger acquisition
        print("  Triggering BRAM acquisition...")
        driver.trigger_acquisition()
        
        # Wait for completion
        timeout = 1.0  # 1 second timeout
        start_time = time.time()
        while not driver.is_acquisition_complete() and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if (time.time() - start_time) >= timeout:
            print("  ⚠️  BRAM acquisition timeout")
            return False, None
        
        # Get data
        print("  Getting BRAM data...")
        adc_data = driver.get_adc_samples_1000()
        dac_data = driver.get_dac_samples_1000()
        
        print(f"  ADC data length: {len(adc_data)}")
        print(f"  DAC data length: {len(dac_data)}")
        
        # Convert to voltage (CORRECTED conversion)
        # ADC: Unsigned 16-bit, 0-3.3V range (NOT ±1.8V!)
        adc_ch0 = (np.array(adc_data, dtype=np.uint32) & 0xFFFF) / 65535.0 * 3.3
        adc_ch1 = ((np.array(adc_data, dtype=np.uint32) >> 16) & 0xFFFF) / 65535.0 * 3.3
        
        print(f"  ADC Ch0 range: {np.min(adc_ch0):.3f}V to {np.max(adc_ch0):.3f}V (mean: {np.mean(adc_ch0):.3f}V)")
        print(f"  ADC Ch1 range: {np.min(adc_ch1):.3f}V to {np.max(adc_ch1):.3f}V (mean: {np.mean(adc_ch1):.3f}V)")
        
        # DAC: Check what we're actually getting
        dac_ch0 = (np.array(dac_data, dtype=np.uint32) & 0xFFFF) / 65535.0 * 2.5
        dac_ch1 = ((np.array(dac_data, dtype=np.uint32) >> 16) & 0xFFFF) / 65535.0 * 2.5
        
        print(f"  DAC Ch0 range: {np.min(dac_ch0):.3f}V to {np.max(dac_ch0):.3f}V (mean: {np.mean(dac_ch0):.3f}V)")
        print(f"  DAC Ch1 range: {np.min(dac_ch1):.3f}V to {np.max(dac_ch1):.3f}V (mean: {np.mean(dac_ch1):.3f}V)")
        
        # Check if data looks reasonable
        ch0_range = np.max(adc_ch0) - np.min(adc_ch0)
        ch1_range = np.max(adc_ch1) - np.min(adc_ch1)
        dac_range = np.max(dac_ch0) - np.min(dac_ch0)
        
        print(f"  Variation - ADC Ch0: {ch0_range:.4f}V, ADC Ch1: {ch1_range:.4f}V, DAC Ch0: {dac_range:.4f}V")
        
        if ch0_range > 0.01 or ch1_range > 0.01:  # More than 10mV range
            print("  ✅ BRAM capture shows good ADC data variation")
        else:
            print("  ⚠️  BRAM ADC data shows little variation")
        
        if dac_range > 0.01:
            print("  ✅ BRAM capture shows DAC variation (ramp working)")
        else:
            print("  ❌ BRAM DAC data is constant (ramp not captured or not working)")
            
        return True, {'adc_ch0': adc_ch0, 'adc_ch1': adc_ch1, 'adc_raw': adc_data, 'dac_raw': dac_data}
            
    except Exception as e:
        print(f"  ❌ BRAM test failed: {e}")
        return False, None

def test_with_ramp_output(driver):
    """Test ADC with known ramp output"""
    print("\n🔄 Testing ADC with Ramp Output...")
    
    try:
        # Set up a slow ramp
        print("  Setting up ramp: 1Hz, 1V amplitude, 1.5V offset")
        driver.set_ramp_frequency(1.0)
        driver.set_ramp_amplitude(1.0)
        driver.set_ramp_offset(1.5)
        
        # Configure the waveform
        print("  Configuring ramp waveform...")
        driver.generate_ramp_waveform()
        time.sleep(0.1)
        
        # Start ramp
        print("  Starting ramp...")
        driver.start_ramp()
        time.sleep(0.5)  # Let it stabilize
        
        # Verify ramp is running
        ramp_enabled = driver.get_ramp_enabled()
        print(f"  Ramp enabled: {ramp_enabled}")
        
        if not ramp_enabled:
            print("  ❌ Failed to start ramp - cannot test loopback")
            return False, None
        
        # Wait a bit longer for the ramp to be generating signal
        print("  Waiting for ramp to generate signal...")
        time.sleep(1.0)
        
        # Test individual readings first
        print("  Testing individual ADC readings with ramp running...")
        for i in range(5):
            voltage = driver.get_photodiode_voltage()
            raw = driver.get_photodiode_raw()
            print(f"    Reading {i+1}: {voltage:.4f}V (raw: {raw})")
            time.sleep(0.2)
        
        # Capture BRAM data
        success, data = test_bram_capture(driver)
        
        # Stop ramp
        print("  Stopping ramp...")
        driver.stop_ramp()
        
        if success and data:
            # Convert DAC data too
            dac_ch0 = (np.array(data['dac_raw'], dtype=np.uint32) & 0xFFFF) / 65535.0 * 2.5
            
            print(f"  DAC output range: {np.min(dac_ch0):.3f}V to {np.max(dac_ch0):.3f}V")
            
            # Check if DAC is actually varying
            dac_range = np.max(dac_ch0) - np.min(dac_ch0)
            if dac_range < 0.1:
                print(f"  ⚠️  DAC output range too small ({dac_range:.3f}V) - ramp may not be active")
                return False, data
            
            # Check correlation with loopback
            if len(data['adc_ch0']) > 0 and len(dac_ch0) > 0:
                correlation = np.corrcoef(dac_ch0, data['adc_ch0'])[0, 1]
                print(f"  Loopback correlation: {correlation:.3f}")
                
                if correlation > 0.7:
                    print("  ✅ Strong correlation - loopback working well!")
                elif correlation > 0.3:
                    print("  ⚠️  Moderate correlation - loopback partially working")
                else:
                    print("  ❌ Poor correlation - check loopback connection")
                
                return True, {**data, 'dac_ch0': dac_ch0, 'correlation': correlation}
        
        return success, data
        
    except Exception as e:
        print(f"  ❌ Ramp test failed: {e}")
        try:
            driver.stop_ramp()  # Ensure ramp is stopped
        except:
            pass
        return False, None

def plot_results(data):
    """Plot the test results"""
    if not data:
        return
    
    print("\n📊 Plotting results...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Time axis (assuming 250 MHz sampling)
    time_us = np.arange(len(data['adc_ch0'])) / 250.0  # microseconds
    
    # ADC Channel 0
    axes[0, 0].plot(time_us, data['adc_ch0'], 'r-', linewidth=1)
    axes[0, 0].set_title('ADC Channel 0')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # ADC Channel 1
    axes[0, 1].plot(time_us, data['adc_ch1'], 'g-', linewidth=1)
    axes[0, 1].set_title('ADC Channel 1')
    axes[0, 1].set_ylabel('Voltage (V)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # If we have DAC data, show comparison
    if 'dac_ch0' in data:
        axes[1, 0].plot(time_us, data['dac_ch0'], 'b-', label='DAC Output', linewidth=2)
        axes[1, 0].plot(time_us, data['adc_ch0'], 'r-', label='ADC Input', linewidth=1, alpha=0.8)
        axes[1, 0].set_title('Loopback Comparison')
        axes[1, 0].set_xlabel('Time (μs)')
        axes[1, 0].set_ylabel('Voltage (V)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # XY plot for correlation
        axes[1, 1].scatter(data['dac_ch0'], data['adc_ch0'], alpha=0.5, s=1)
        axes[1, 1].set_xlabel('DAC Output (V)')
        axes[1, 1].set_ylabel('ADC Input (V)')
        axes[1, 1].set_title(f'Correlation Plot (r={data.get("correlation", 0):.3f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add ideal line
        min_val = min(np.min(data['dac_ch0']), np.min(data['adc_ch0']))
        max_val = max(np.max(data['dac_ch0']), np.max(data['adc_ch0']))
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal')
        axes[1, 1].legend()
    else:
        # Just show raw data histograms
        axes[1, 0].hist(data['adc_ch0'], bins=50, alpha=0.7, color='red', label='Ch0')
        axes[1, 0].set_title('ADC Ch0 Distribution')
        axes[1, 0].set_xlabel('Voltage (V)')
        axes[1, 0].set_ylabel('Count')
        
        axes[1, 1].hist(data['adc_ch1'], bins=50, alpha=0.7, color='green', label='Ch1')
        axes[1, 1].set_title('ADC Ch1 Distribution')
        axes[1, 1].set_xlabel('Voltage (V)')
        axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""
    print("🔬 Simple ADC Input Test")
    print("========================")
    
    # Connect to device
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        print(f"Connecting to {host}...")
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Run tests
    print("\n" + "="*50)
    
    # Test 1: Basic ADC readings
    test_adc_readings(driver)
    
    # Test 2: BRAM capture without ramp
    print("\n" + "="*50)
    bram_success, bram_data = test_bram_capture(driver)
    
    # Test 3: BRAM capture with ramp (loopback test)
    print("\n" + "="*50)
    ramp_success, ramp_data = test_with_ramp_output(driver)
    
    # Plot results
    if ramp_success and ramp_data:
        plot_results(ramp_data)
    elif bram_success and bram_data:
        plot_results(bram_data)
    
    # Summary
    print("\n" + "="*50)
    print("📋 Test Summary:")
    print(f"  BRAM Capture: {'✅ Working' if bram_success else '❌ Failed'}")
    print(f"  Ramp+ADC Test: {'✅ Working' if ramp_success else '❌ Failed'}")
    
    if ramp_success:
        print("🎉 ADC input appears to be working correctly!")
        print("   You can now proceed with implementing the live web interface.")
    elif bram_success:
        print("⚠️  BRAM capture works, but check your loopback connection.")
    else:
        print("❌ ADC input needs debugging. Check hardware connections.")

if __name__ == "__main__":
    main() 