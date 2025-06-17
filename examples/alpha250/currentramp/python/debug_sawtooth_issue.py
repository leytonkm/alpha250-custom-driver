#!/usr/bin/env python3
"""
Debug Sawtooth vs Triangle Wave Issue
Diagnose why we're seeing triangle instead of sawtooth and flat ADC
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
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
    def start_ramp(self):
        pass
    
    @command()
    def stop_ramp(self):
        pass
    
    @command()
    def get_ramp_enabled(self):
        return self.client.recv_bool()
    
    @command()
    def get_ramp_phase(self):
        return self.client.recv_uint32()
    
    @command()
    def get_cycle_count(self):
        return self.client.recv_uint32()
    
    @command()
    def get_ramp_position(self):
        return self.client.recv_float()
    
    @command()
    def get_photodiode_voltage(self):
        return self.client.recv_float()
    
    @command()
    def get_photodiode_raw(self):
        return self.client.recv_int16()
    
    @command()
    def set_photodiode_channel(self, channel):
        pass

def convert_bram_data_corrected(adc_raw, dac_raw):
    """Convert raw BRAM data with corrected format"""
    fs = 250e6  # 250 MHz sampling rate
    n_samples = min(len(adc_raw), len(dac_raw))
    
    # Time base
    time_us = np.arange(n_samples) / fs * 1e6  # microseconds
    
    # DAC conversion: Hardware ramp output (ACTUAL sawtooth from FPGA)
    dac_voltages = (np.array(dac_raw[:n_samples]) & 0xFFFF) / 65535.0 * 2.5
    
    # ADC conversion: Raw ADC data from LTC2157
    # Lower 16 bits = ADC channel 0, Upper 16 bits = ADC channel 1
    adc_ch0_raw = np.array(adc_raw[:n_samples]) & 0xFFFF
    adc_ch1_raw = (np.array(adc_raw[:n_samples]) >> 16) & 0xFFFF
    
    # Convert ADC channel 0 (photodiode input)
    adc_signed = np.where(adc_ch0_raw > 32767, adc_ch0_raw - 65536, adc_ch0_raw)
    adc_voltages = (adc_signed / 32768.0) * 1.8
    
    return time_us, dac_voltages, adc_voltages, adc_ch0_raw, adc_ch1_raw

def debug_sawtooth_issue():
    """Debug the sawtooth vs triangle wave issue"""
    print("🔍 DEBUGGING SAWTOOTH vs TRIANGLE ISSUE")
    print("=" * 60)
    
    # Connect
    host = os.environ.get('HOST', '192.168.1.20')
    client = connect(host, 'currentramp', restart=False)
    driver = CurrentRamp(client)
    
    print(f"Connected to {host}")
    
    # Configure ramp for testing
    print("\n⚙️  Configuring ramp for debugging...")
    driver.set_ramp_frequency(10.0)  # 10 Hz - should see 10 cycles per second
    driver.set_ramp_amplitude(1.0)   # 1V amplitude
    driver.set_ramp_offset(1.5)      # 1.5V offset
    driver.generate_ramp_waveform()
    driver.start_ramp()
    time.sleep(0.5)  # Let ramp stabilize
    
    enabled = driver.get_ramp_enabled()
    print(f"  Ramp enabled: {enabled}")
    
    if not enabled:
        print("❌ Ramp not enabled - cannot debug")
        return
    
    # Check hardware status
    print("\n📊 Hardware Status Check:")
    phase = driver.get_ramp_phase()
    cycles = driver.get_cycle_count()
    position = driver.get_ramp_position()
    
    print(f"  Phase accumulator: 0x{phase:08X}")
    print(f"  Cycle count: {cycles}")
    print(f"  Ramp position: {position:.4f}")
    
    # Monitor for a few cycles to see if hardware is working
    print("\n⏱️  Monitoring hardware for 2 seconds...")
    start_cycles = cycles
    time.sleep(2.0)
    
    final_cycles = driver.get_cycle_count()
    cycles_elapsed = final_cycles - start_cycles
    expected_cycles = 2.0 * 10.0  # 2 seconds * 10 Hz
    
    print(f"  Cycles elapsed: {cycles_elapsed} (expected ~{expected_cycles:.0f})")
    
    if cycles_elapsed < expected_cycles * 0.8:
        print("⚠️  WARNING: Hardware ramp frequency may be incorrect")
    else:
        print("✅ Hardware ramp timing looks correct")
    
    # Test individual ADC readings
    print("\n📡 Testing Individual ADC Readings:")
    driver.set_photodiode_channel(0)
    time.sleep(0.1)
    
    adc_readings = []
    for i in range(10):
        voltage = driver.get_photodiode_voltage()
        raw = driver.get_photodiode_raw()
        adc_readings.append(voltage)
        print(f"  Reading {i+1}: {voltage:.4f}V (raw: {raw})")
        time.sleep(0.1)
    
    adc_variation = max(adc_readings) - min(adc_readings)
    print(f"  ADC variation: {adc_variation:.4f}V")
    
    if adc_variation < 0.001:
        print("❌ ADC shows no variation - check input signal connection")
    else:
        print("✅ ADC shows variation - input signal detected")
    
    # Perform BRAM acquisition
    print("\n⚡ BRAM Data Acquisition:")
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
    
    # Get and analyze data
    adc_data = driver.get_adc_samples_1000()
    dac_data = driver.get_dac_samples_1000()
    
    time_us, dac_voltages, adc_voltages, adc_ch0_raw, adc_ch1_raw = convert_bram_data_corrected(adc_data, dac_data)
    
    # Analyze DAC waveform
    print(f"\n📈 DAC Waveform Analysis:")
    print(f"  Time span: {time_us[-1]-time_us[0]:.2f} μs")
    print(f"  DAC range: {np.min(dac_voltages):.4f}V to {np.max(dac_voltages):.4f}V")
    print(f"  DAC mean: {np.mean(dac_voltages):.4f}V")
    print(f"  DAC std: {np.std(dac_voltages):.4f}V")
    
    # Check for sawtooth characteristics
    dac_diff = np.diff(dac_voltages)
    positive_steps = np.sum(dac_diff > 0)
    negative_steps = np.sum(dac_diff < 0)
    
    print(f"  Positive steps: {positive_steps}")
    print(f"  Negative steps: {negative_steps}")
    
    if negative_steps < positive_steps * 0.1:
        print("✅ DAC shows sawtooth characteristics (mostly positive steps)")
    elif abs(positive_steps - negative_steps) < len(dac_diff) * 0.2:
        print("⚠️  DAC shows triangle characteristics (equal positive/negative steps)")
    else:
        print("❓ DAC waveform unclear")
    
    # Analyze ADC waveform
    print(f"\n📡 ADC Waveform Analysis:")
    print(f"  ADC range: {np.min(adc_voltages):.4f}V to {np.max(adc_voltages):.4f}V")
    print(f"  ADC mean: {np.mean(adc_voltages):.4f}V")
    print(f"  ADC std: {np.std(adc_voltages):.4f}V")
    print(f"  ADC Ch0 raw range: {np.min(adc_ch0_raw)} to {np.max(adc_ch0_raw)}")
    print(f"  ADC Ch1 raw range: {np.min(adc_ch1_raw)} to {np.max(adc_ch1_raw)}")
    
    # Create diagnostic plots
    print(f"\n📊 Creating diagnostic plots...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: DAC waveform
    ax1.plot(time_us, dac_voltages, 'b-', linewidth=1)
    ax1.set_title('DAC Output (Hardware Ramp)')
    ax1.set_xlabel('Time (μs)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ADC waveform
    ax2.plot(time_us, adc_voltages, 'r-', linewidth=1)
    ax2.set_title('ADC Input (Channel 0)')
    ax2.set_xlabel('Time (μs)')
    ax2.set_ylabel('Voltage (V)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Raw ADC data
    ax3.plot(time_us, adc_ch0_raw, 'g-', linewidth=1, label='Ch0 Raw')
    ax3.plot(time_us, adc_ch1_raw, 'm-', linewidth=1, label='Ch1 Raw')
    ax3.set_title('Raw ADC Data')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Raw Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: DAC derivative (to show sawtooth vs triangle)
    ax4.plot(time_us[1:], np.diff(dac_voltages), 'b-', linewidth=1)
    ax4.set_title('DAC Derivative (Sawtooth Check)')
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('dV/dt')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sawtooth_debug.png', dpi=300, bbox_inches='tight')
    print(f"✅ Diagnostic plots saved as 'sawtooth_debug.png'")
    
    # Final diagnosis
    print(f"\n🎯 DIAGNOSIS:")
    
    dac_variation = np.max(dac_voltages) - np.min(dac_voltages)
    adc_variation = np.max(adc_voltages) - np.min(adc_voltages)
    
    if dac_variation < 0.1:
        print("❌ DAC Problem: Little variation - ramp not working")
    elif negative_steps > positive_steps * 0.5:
        print("❌ DAC Problem: Triangle wave instead of sawtooth")
        print("   → Check FPGA ramp generator configuration")
    else:
        print("✅ DAC: Sawtooth working correctly")
    
    if adc_variation < 0.01:
        print("❌ ADC Problem: No input signal detected")
        print("   → Check sine wave connection to ADC channel 0")
        print("   → Verify signal amplitude and frequency")
    else:
        print("✅ ADC: Input signal detected")
    
    # Stop ramp
    driver.stop_ramp()
    
    return True

if __name__ == "__main__":
    try:
        debug_sawtooth_issue()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the device is connected and FPGA is programmed") 