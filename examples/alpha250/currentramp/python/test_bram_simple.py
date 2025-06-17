#!/usr/bin/env python3
"""
Simple BRAM Test - Verify Data Format
"""

import os
import time
import numpy as np
from koheron import connect, command

class CurrentRamp:
    def __init__(self, client):
        self.client = client

    @command()
    def trigger_acquisition(self):
        pass
    
    @command()
    def is_acquisition_complete(self):
        return self.client.recv_bool()
    
    @command()
    def get_adc_samples_1000(self):
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def get_dac_samples_1000(self):
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def start_ramp(self):
        pass
    
    @command()
    def stop_ramp(self):
        pass
    
    @command()
    def set_ramp_frequency(self, frequency):
        pass
    
    @command()
    def set_ramp_amplitude(self, amplitude):
        pass
    
    @command()
    def set_ramp_offset(self, offset):
        pass
    
    @command()
    def generate_ramp_waveform(self):
        pass
    
    @command()
    def get_ramp_enabled(self):
        return self.client.recv_bool()

    @command()
    def get_dac_channel(self, channel):
        return self.client.recv_float()

    @command()
    def set_test_voltage_channel_2(self, voltage):
        pass

def convert_bram_data(adc_raw, dac_raw):
    """Convert raw BRAM data to voltages with correct timing"""
    fs = 250e6  # 250 MHz sampling rate (full speed, no decimation)
    n_samples = min(len(adc_raw), len(dac_raw))
    
    # Time base: sample index / sampling frequency
    time_us = np.arange(n_samples) / fs * 1e6  # microseconds
    
    # DAC conversion: Precision DAC data (32-bit format)
    # Lower 16 bits = DAC channel 2 (ramp), Upper 16 bits = DAC channel 3 (unused)
    dac_ch2_raw = np.array(dac_raw[:n_samples]) & 0xFFFF
    dac_ch3_raw = (np.array(dac_raw[:n_samples]) >> 16) & 0xFFFF
    
    # DAC channel 2 is the ramp output (16-bit unsigned, 0-2.5V range)
    dac_voltages = dac_ch2_raw / 65535.0 * 2.5
    
    # ADC conversion: Raw ADC data from fast ADC (LTC2157)
    # Lower 16 bits = ADC channel 0, Upper 16 bits = ADC channel 1
    adc_ch0_raw = np.array(adc_raw[:n_samples]) & 0xFFFF
    adc_ch1_raw = (np.array(adc_raw[:n_samples]) >> 16) & 0xFFFF
    
    # Use ADC channel 0 (your input signal)
    # LTC2157: 16-bit signed, ±1.8V range (Alpha250 standard)
    adc_ch0_signed = np.where(adc_ch0_raw > 32767, adc_ch0_raw - 65536, adc_ch0_raw)
    adc_ch0_voltages = (adc_ch0_signed / 32768.0) * 1.8
    
    # ADC channel 1 for comparison
    adc_ch1_signed = np.where(adc_ch1_raw > 32767, adc_ch1_raw - 65536, adc_ch1_raw)
    adc_ch1_voltages = (adc_ch1_signed / 32768.0) * 1.8
    
    return time_us, dac_voltages, adc_ch0_voltages, adc_ch1_voltages

def test_bram_simple():
    """Simple BRAM test"""
    print("🔍 SIMPLE BRAM TEST")
    print("=" * 40)
    
    # Connect
    host = os.environ.get('HOST', '192.168.1.20')
    client = connect(host, 'currentramp', restart=False)
    driver = CurrentRamp(client)
    
    print(f"Connected to {host}")
    
    # Start ramp with HIGH FREQUENCY to see variation in 4us window
    print("\n⚙️  Starting HIGH FREQUENCY ramp...")
    driver.set_ramp_frequency(1000000.0)  # 1 MHz ramp - should show ~4 cycles in 4us
    driver.set_ramp_amplitude(1.0)
    driver.set_ramp_offset(1.5)
    driver.generate_ramp_waveform()
    driver.start_ramp()
    time.sleep(0.5)
    
    enabled = driver.get_ramp_enabled()
    print(f"  Ramp enabled: {enabled}")
    
    # Trigger BRAM acquisition
    print("\n📊 Triggering BRAM acquisition...")
    driver.trigger_acquisition()
    
    # Wait for completion
    timeout = 1.0
    start_time = time.time()
    while not driver.is_acquisition_complete() and (time.time() - start_time) < timeout:
        time.sleep(0.001)
    
    if (time.time() - start_time) >= timeout:
        print("❌ BRAM acquisition timeout")
        return
    
    print("✅ BRAM acquisition completed")
    
    # Get raw data
    adc_data = driver.get_adc_samples_1000()
    dac_data = driver.get_dac_samples_1000()
    
    print(f"\n📈 Raw Data Analysis:")
    print(f"  ADC samples: {len(adc_data)}")
    print(f"  DAC samples: {len(dac_data)}")
    
    # Show first 10 raw samples
    print(f"\n🔬 First 10 Raw Samples:")
    print("  Index | ADC_Raw     | DAC_Raw     | ADC_Ch0 | ADC_Ch1 | DAC_Val")
    print("  ------|-------------|-------------|---------|---------|--------")
    
    for i in range(min(10, len(adc_data), len(dac_data))):
        adc_raw = adc_data[i]
        dac_raw = dac_data[i]
        
        # Extract channels
        adc_ch0 = adc_raw & 0xFFFF
        adc_ch1 = (adc_raw >> 16) & 0xFFFF
        dac_val = dac_raw & 0xFFFF
        
        print(f"  {i:5d} | 0x{adc_raw:08X} | 0x{dac_raw:08X} | {adc_ch0:7d} | {adc_ch1:7d} | {dac_val:7d}")
    
    # Check for variation
    adc_ch0_all = np.array([d & 0xFFFF for d in adc_data])
    adc_ch1_all = np.array([(d >> 16) & 0xFFFF for d in adc_data])
    dac_all = np.array([d & 0xFFFF for d in dac_data])
    
    print(f"\n📊 Variation Analysis:")
    print(f"  ADC Ch0: min={np.min(adc_ch0_all)}, max={np.max(adc_ch0_all)}, range={np.max(adc_ch0_all)-np.min(adc_ch0_all)}")
    print(f"  ADC Ch1: min={np.min(adc_ch1_all)}, max={np.max(adc_ch1_all)}, range={np.max(adc_ch1_all)-np.min(adc_ch1_all)}")
    print(f"  DAC:     min={np.min(dac_all)}, max={np.max(dac_all)}, range={np.max(dac_all)-np.min(dac_all)}")
    
    # Convert to voltages with correct timing
    print(f"\n🔧 Voltage Conversion (NEW - 10 kHz Decimated Sampling):")
    time_us, dac_voltages, adc_ch0_voltages, adc_ch1_voltages = convert_bram_data(adc_data, dac_data)
    
    print(f"  Time span: {time_us[0]:.1f}us to {time_us[-1]:.1f}us ({time_us[-1]-time_us[0]:.1f}us total)")
    print(f"  DAC Ch2: {np.min(dac_voltages):.4f}V to {np.max(dac_voltages):.4f}V")
    print(f"  ADC Ch0: {np.min(adc_ch0_voltages):.4f}V to {np.max(adc_ch0_voltages):.4f}V")
    print(f"  ADC Ch1: {np.min(adc_ch1_voltages):.4f}V to {np.max(adc_ch1_voltages):.4f}V")
    
    # Check if data looks reasonable for 10 Hz ramp
    dac_range = np.max(dac_voltages) - np.min(dac_voltages)
    adc_ch0_range = np.max(adc_ch0_voltages) - np.min(adc_ch0_voltages)
    adc_ch1_range = np.max(adc_ch1_voltages) - np.min(adc_ch1_voltages)
    
    print(f"\n🎯 DIAGNOSIS (1 MHz Ramp Analysis):")
    
    # For 1 MHz ramp, 4us window should show ~4 full cycles
    expected_cycles = (time_us[-1] - time_us[0]) / 1.0  # 1us per cycle at 1 MHz
    print(f"  Expected ramp cycles in {time_us[-1]-time_us[0]:.1f}us window: {expected_cycles:.1f}")
    
    if dac_range > 0.5:  # Should see significant ramp variation
        print("✅ DAC Ch2 shows good ramp variation - WORKING!")
    elif dac_range > 0.1:
        print("⚠️  DAC Ch2 shows some variation - partial ramp")
    else:
        print("❌ DAC Ch2 shows little variation - ramp not working")
    
    if adc_ch0_range > 0.05:  # Should see your input signal
        print("✅ ADC Ch0 shows good signal variation - INPUT DETECTED!")
    elif adc_ch0_range > 0.01:
        print("⚠️  ADC Ch0 shows small variation - weak signal")
    else:
        print("❌ ADC Ch0 flat - no input signal")
    
    if adc_ch1_range > 0.01:
        print("✅ ADC Ch1 shows variation")
    else:
        print("❌ ADC Ch1 flat - no signal")
    
    # Check precision DAC status by setting test voltages
    print(f"\n🔧 Precision DAC Test:")
    try:
        # Test DAC channel 2 with different voltages
        print(f"  Testing DAC channel 2 with different voltages...")
        
        # Set DAC channel 2 to 1.0V
        driver.set_test_voltage_channel_2(1.0)
        time.sleep(0.1)
        
        # Trigger BRAM acquisition
        driver.trigger_acquisition()
        while not driver.is_acquisition_complete():
            time.sleep(0.001)
        
        # Get data
        dac_data_1v = driver.get_dac_samples_1000()
        dac_val_1v = dac_data_1v[0] & 0xFFFF
        voltage_1v = (dac_val_1v / 65535.0) * 2.5
        print(f"  DAC Ch2 @ 1.0V: Raw={dac_val_1v}, Voltage={voltage_1v:.4f}V")
        
        # Set DAC channel 2 to 2.0V
        driver.set_test_voltage_channel_2(2.0)
        time.sleep(0.1)
        
        # Trigger BRAM acquisition
        driver.trigger_acquisition()
        while not driver.is_acquisition_complete():
            time.sleep(0.001)
        
        # Get data
        dac_data_2v = driver.get_dac_samples_1000()
        dac_val_2v = dac_data_2v[0] & 0xFFFF
        voltage_2v = (dac_val_2v / 65535.0) * 2.5
        print(f"  DAC Ch2 @ 2.0V: Raw={dac_val_2v}, Voltage={voltage_2v:.4f}V")
        
        # Check if BRAM is capturing DAC changes
        if abs(voltage_2v - voltage_1v) > 0.5:
            print("✅ BRAM is capturing DAC channel 2 changes!")
        else:
            print("❌ BRAM is NOT capturing DAC channel 2 changes!")
            
    except Exception as e:
        print(f"  Error testing DAC: {e}")
    
    # Test with actual ramp running - multiple acquisitions
    print(f"\n🚀 RAMP VARIATION TEST:")
    try:
        print(f"  Testing ramp variation with multiple BRAM captures...")
        
        # Start ramp again
        driver.start_ramp()
        time.sleep(0.1)
        
        # Capture multiple BRAM acquisitions to see ramp progression
        ramp_values = []
        for i in range(5):
            driver.trigger_acquisition()
            while not driver.is_acquisition_complete():
                time.sleep(0.001)
            
            dac_data = driver.get_dac_samples_1000()
            dac_val = dac_data[0] & 0xFFFF
            voltage = (dac_val / 65535.0) * 2.5
            ramp_values.append(voltage)
            print(f"  Capture {i+1}: Raw={dac_val}, Voltage={voltage:.4f}V")
            
            time.sleep(0.01)  # 10ms between captures
        
        # Check if ramp is progressing
        ramp_range = max(ramp_values) - min(ramp_values)
        print(f"  Ramp variation over 50ms: {ramp_range:.4f}V")
        
        if ramp_range > 0.01:
            print("✅ Ramp is working - variation detected over time!")
        else:
            print("❌ Ramp is static - no variation over time!")
            
    except Exception as e:
        print(f"  Error testing ramp: {e}")
    
    # Stop ramp
    driver.stop_ramp()
    
    return True

if __name__ == "__main__":
    try:
        test_bram_simple()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the device is connected and FPGA is programmed") 