#!/usr/bin/env python3
"""
Debug Raw BRAM Data
Quick diagnostic to understand the actual data format
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
    def get_adc_samples_1000(self):
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def get_dac_samples_1000(self):
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def is_acquisition_complete(self):
        return self.client.recv_bool()
    
    @command()
    def get_photodiode_voltage(self):
        return self.client.recv_float()
    
    @command()
    def get_photodiode_raw(self):
        return self.client.recv_int16()
    
    @command()
    def set_photodiode_channel(self, channel):
        pass
    
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

def analyze_raw_data():
    print("🔍 Raw BRAM Data Analysis")
    print("=" * 50)
    
    # Connect
    host = os.environ.get('HOST', '192.168.1.20')
    client = connect(host, 'currentramp', restart=False)
    driver = CurrentRamp(client)
    
    # Get individual ADC readings for reference
    print("\n📊 Individual ADC Readings (for reference):")
    for ch in [0, 1]:
        driver.set_photodiode_channel(ch)
        time.sleep(0.1)
        voltage = driver.get_photodiode_voltage()
        raw = driver.get_photodiode_raw()
        print(f"  ADC Ch{ch}: {voltage:.4f}V (raw: {raw}, hex: 0x{raw & 0xFFFF:04X})")
    
    # Start ramp for comparison
    print("\n🔄 Starting ramp for comparison...")
    driver.set_ramp_frequency(10.0)
    driver.set_ramp_amplitude(1.0) 
    driver.set_ramp_offset(1.5)
    driver.generate_ramp_waveform()
    driver.start_ramp()
    time.sleep(0.5)
    
    enabled = driver.get_ramp_enabled()
    print(f"  Ramp enabled: {enabled}")
    
    # Get BRAM data
    print("\n📈 BRAM Data Analysis:")
    print("  Triggering acquisition...")
    driver.trigger_acquisition()
    time.sleep(0.1)
    
    adc_data = driver.get_adc_samples_1000()
    dac_data = driver.get_dac_samples_1000()
    
    print(f"  ADC data length: {len(adc_data)}")
    print(f"  DAC data length: {len(dac_data)}")
    
    # Analyze first few samples
    print("\n🔬 First 10 Raw Samples:")
    print("  Index | ADC_Raw     | Ch0_Raw | Ch1_Raw | DAC_Raw     | Ch0_DAC | Ch1_DAC")
    print("  ------|-------------|---------|---------|-------------|---------|--------")
    
    for i in range(10):
        adc_raw = adc_data[i]
        dac_raw = dac_data[i]
        
        # Extract channels (assuming concatenated format)
        adc_ch0_raw = adc_raw & 0xFFFF
        adc_ch1_raw = (adc_raw >> 16) & 0xFFFF
        dac_ch0_raw = dac_raw & 0xFFFF
        dac_ch1_raw = (dac_raw >> 16) & 0xFFFF
        
        print(f"  {i:5d} | 0x{adc_raw:08X} | {adc_ch0_raw:7d} | {adc_ch1_raw:7d} | 0x{dac_raw:08X} | {dac_ch0_raw:7d} | {dac_ch1_raw:7d}")
    
    # Statistical analysis
    print(f"\n📊 Statistical Analysis (1000 samples):")
    
    adc_ch0_all = adc_data & 0xFFFF
    adc_ch1_all = (adc_data >> 16) & 0xFFFF
    dac_ch0_all = dac_data & 0xFFFF
    dac_ch1_all = (dac_data >> 16) & 0xFFFF
    
    print(f"  ADC Ch0: min={np.min(adc_ch0_all)}, max={np.max(adc_ch0_all)}, mean={np.mean(adc_ch0_all):.1f}, std={np.std(adc_ch0_all):.1f}")
    print(f"  ADC Ch1: min={np.min(adc_ch1_all)}, max={np.max(adc_ch1_all)}, mean={np.mean(adc_ch1_all):.1f}, std={np.std(adc_ch1_all):.1f}")
    print(f"  DAC Ch0: min={np.min(dac_ch0_all)}, max={np.max(dac_ch0_all)}, mean={np.mean(dac_ch0_all):.1f}, std={np.std(dac_ch0_all):.1f}")
    print(f"  DAC Ch1: min={np.min(dac_ch1_all)}, max={np.max(dac_ch1_all)}, mean={np.mean(dac_ch1_all):.1f}, std={np.std(dac_ch1_all):.1f}")
    
    # Voltage conversion attempts
    print(f"\n🔧 Voltage Conversion Attempts:")
    
    # Method 1: Alpha250 standard (signed, ±1.8V)
    adc_ch0_v1 = ((adc_ch0_all.astype(np.int32) - 32768) / 32768.0) * 1.8
    print(f"  Method 1 (±1.8V): Ch0 = {np.mean(adc_ch0_v1):.4f}V ± {np.std(adc_ch0_v1):.4f}V")
    
    # Method 2: Unsigned, 0-3.3V
    adc_ch0_v2 = (adc_ch0_all / 65535.0) * 3.3
    print(f"  Method 2 (0-3.3V): Ch0 = {np.mean(adc_ch0_v2):.4f}V ± {np.std(adc_ch0_v2):.4f}V")
    
    # Method 3: Two's complement interpretation
    adc_ch0_signed = adc_ch0_all.astype(np.int16)
    adc_ch0_v3 = (adc_ch0_signed / 32768.0) * 1.8
    print(f"  Method 3 (Two's complement): Ch0 = {np.mean(adc_ch0_v3):.4f}V ± {np.std(adc_ch0_v3):.4f}V")
    
    # Method 4: Check if data is actually different format
    if np.std(adc_ch0_all) < 100:
        print(f"  ⚠️  Very low variation suggests constant data or wrong interpretation")
    
    # Expected individual reading comparison
    driver.set_photodiode_channel(0)
    time.sleep(0.1)
    individual_voltage = driver.get_photodiode_voltage()
    individual_raw = driver.get_photodiode_raw()
    
    print(f"\n🎯 Individual vs BRAM Comparison:")
    print(f"  Individual reading: {individual_voltage:.4f}V (raw: {individual_raw})")
    print(f"  BRAM Ch0 raw mean: {np.mean(adc_ch0_all):.1f}")
    print(f"  Raw value ratio: {np.mean(adc_ch0_all) / (individual_raw & 0xFFFF):.3f}")
    
    # Stop ramp
    driver.stop_ramp()
    
    print(f"\n💡 Diagnosis:")
    expected_raw = individual_raw & 0xFFFF
    bram_raw = np.mean(adc_ch0_all)
    
    if abs(bram_raw - expected_raw) < 1000:
        print(f"  ✅ BRAM and individual readings are similar - format is likely correct")
    else:
        print(f"  ❌ BRAM and individual readings differ significantly")
        print(f"     Expected raw: ~{expected_raw}")
        print(f"     BRAM raw: ~{bram_raw:.0f}")
        print(f"     This suggests BRAM timing or trigger issues")

if __name__ == "__main__":
    analyze_raw_data() 