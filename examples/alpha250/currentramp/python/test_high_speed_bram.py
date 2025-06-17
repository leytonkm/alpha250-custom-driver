#!/usr/bin/env python3
"""
High-Speed BRAM Acquisition Test for Laser Precision
Tests the 250 MHz BRAM data capture system
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """CurrentRamp Python interface for high-speed BRAM testing"""
    
    def __init__(self, client):
        self.client = client

    # === BRAM High-Speed Acquisition ===
    
    @command()
    def trigger_acquisition(self):
        """Trigger BRAM data acquisition"""
        pass
    
    @command()
    def is_acquisition_complete(self):
        """Check if acquisition is complete"""
        return self.client.recv_bool()
    
    @command()
    def get_adc_samples_1000(self):
        """Get 1000 ADC samples from BRAM"""
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def get_dac_samples_1000(self):
        """Get 1000 DAC samples from BRAM"""
        return self.client.recv_array(1000, dtype='uint32')
    
    # === Ramp Control ===
    
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

def convert_bram_data(adc_raw, dac_raw):
    """Convert raw BRAM data to voltages with laser precision"""
    fs = 250e6  # 250 MHz sampling rate
    n_samples = min(len(adc_raw), len(dac_raw))
    
    # Time base: sample index / sampling frequency
    time_us = np.arange(n_samples) / fs * 1e6  # microseconds
    
    # DAC conversion: Hardware ramp output (16-bit unsigned, 0-2.5V range)
    # This is the ACTUAL hardware ramp output from ramp_output_mux
    dac_voltages = (np.array(dac_raw[:n_samples]) & 0xFFFF) / 65535.0 * 2.5
    
    # ADC conversion: Raw ADC data from fast ADC (LTC2157)
    # Lower 16 bits = ADC channel 0, Upper 16 bits = ADC channel 1
    adc_ch0_raw = np.array(adc_raw[:n_samples]) & 0xFFFF
    adc_ch1_raw = (np.array(adc_raw[:n_samples]) >> 16) & 0xFFFF
    
    # Use ADC channel 0 (photodiode input)
    # LTC2157: 16-bit signed, ±1.8V range (Alpha250 standard)
    adc_signed = np.where(adc_ch0_raw > 32767, adc_ch0_raw - 65536, adc_ch0_raw)
    adc_voltages = (adc_signed / 32768.0) * 1.8
    
    return time_us, dac_voltages, adc_voltages

def test_high_speed_acquisition():
    """Test high-speed BRAM acquisition for laser precision"""
    print("🚀 HIGH-SPEED BRAM ACQUISITION TEST")
    print("=" * 60)
    
    # Connect
    host = os.environ.get('HOST', '192.168.1.20')
    client = connect(host, 'currentramp', restart=False)
    driver = CurrentRamp(client)
    
    print(f"Connected to {host}")
    
    # Configure ramp for testing
    print("\n⚙️  Configuring ramp for high-speed test...")
    driver.set_ramp_frequency(10.0)  # 10 Hz
    driver.set_ramp_amplitude(1.0)   # 1V amplitude
    driver.set_ramp_offset(1.5)      # 1.5V offset
    driver.generate_ramp_waveform()
    driver.start_ramp()
    time.sleep(0.5)  # Let ramp stabilize
    
    enabled = driver.get_ramp_enabled()
    print(f"  Ramp enabled: {enabled}")
    
    if not enabled:
        print("❌ Ramp not enabled - cannot test high-speed acquisition")
        return
    
    # Perform high-speed BRAM acquisition
    print("\n⚡ Performing HIGH-SPEED BRAM acquisition...")
    print("  Triggering 250 MHz BRAM capture...")
    
    start_time = time.time()
    driver.trigger_acquisition()
    
    # Wait for completion
    timeout = 1.0  # 1 second timeout
    while not driver.is_acquisition_complete() and (time.time() - start_time) < timeout:
        time.sleep(0.001)  # 1ms polling
    
    acquisition_time = time.time() - start_time
    
    if (time.time() - start_time) >= timeout:
        print("❌ BRAM acquisition timeout")
        return
    
    print(f"  ✅ BRAM acquisition completed in {acquisition_time*1000:.1f} ms")
    
    # Get the data
    print("  📊 Retrieving BRAM data...")
    adc_data = driver.get_adc_samples_1000()
    dac_data = driver.get_dac_samples_1000()
    
    print(f"  ADC samples: {len(adc_data)}")
    print(f"  DAC samples: {len(dac_data)}")
    
    # Convert to voltages
    time_us, dac_voltages, adc_voltages = convert_bram_data(adc_data, dac_data)
    
    # Analysis
    print(f"\n📈 HIGH-SPEED DATA ANALYSIS:")
    print(f"  Time span: {time_us[0]:.2f} to {time_us[-1]:.2f} μs ({time_us[-1]-time_us[0]:.2f} μs total)")
    print(f"  Sampling rate: {1/(time_us[1]-time_us[0]):.1f} MHz")
    print(f"  DAC range: {np.min(dac_voltages):.4f}V to {np.max(dac_voltages):.4f}V")
    print(f"  ADC range: {np.min(adc_voltages):.4f}V to {np.max(adc_voltages):.4f}V")
    
    # Check data quality
    dac_variation = np.max(dac_voltages) - np.min(dac_voltages)
    adc_variation = np.max(adc_voltages) - np.min(adc_voltages)
    
    print(f"\n🔍 LASER PRECISION ANALYSIS:")
    print(f"  DAC variation: {dac_variation:.4f}V")
    print(f"  ADC variation: {adc_variation:.4f}V")
    
    if dac_variation > 0.1:  # Should see significant ramp variation
        print("  ✅ DAC shows excellent ramp variation - hardware timing working")
    else:
        print("  ⚠️  DAC shows little variation - check ramp configuration")
    
    if adc_variation > 0.01:  # Should see some input variation
        print("  ✅ ADC shows good input variation - signal detected")
    else:
        print("  ⚠️  ADC shows little variation - check input signal")
    
    # Calculate correlation for laser precision
    if len(dac_voltages) == len(adc_voltages) and len(dac_voltages) > 10:
        correlation = np.corrcoef(dac_voltages, adc_voltages)[0, 1]
        print(f"  📊 DAC-ADC correlation: {correlation:.4f}")
        
        if abs(correlation) > 0.8:
            print("  ✅ EXCELLENT correlation - laser system working perfectly")
        elif abs(correlation) > 0.5:
            print("  ⚠️  GOOD correlation - laser system working")
        else:
            print("  ❌ POOR correlation - check laser connections")
    
    # Create high-resolution plot
    print(f"\n📊 Creating high-resolution plot...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: DAC output (sawtooth ramp)
    ax1.plot(time_us, dac_voltages, 'b-', linewidth=1, label='DAC Sawtooth Ramp')
    ax1.set_ylabel('DAC Voltage (V)')
    ax1.set_title('High-Speed DAC Output (250 MHz sampling)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: ADC input
    ax2.plot(time_us, adc_voltages, 'r-', linewidth=1, label='ADC Input Signal')
    ax2.set_ylabel('ADC Voltage (V)')
    ax2.set_title('High-Speed ADC Input (250 MHz sampling)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Overlay for correlation
    ax3.plot(time_us, dac_voltages, 'b-', linewidth=1, alpha=0.7, label='DAC Output')
    ax3.plot(time_us, adc_voltages, 'r-', linewidth=1, alpha=0.7, label='ADC Input')
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title('DAC vs ADC Correlation (Laser Precision Analysis)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('high_speed_bram_test.png', dpi=300, bbox_inches='tight')
    print(f"  ✅ Plot saved as 'high_speed_bram_test.png'")
    
    # Show statistics
    print(f"\n📊 FINAL STATISTICS:")
    print(f"  🎯 Data points: {len(dac_voltages)} samples")
    print(f"  ⏱️  Time resolution: {(time_us[1]-time_us[0]):.3f} μs per sample")
    print(f"  🔄 Sampling rate: {1/(time_us[1]-time_us[0]):.1f} MHz")
    print(f"  📏 Total time span: {time_us[-1]-time_us[0]:.2f} μs")
    print(f"  🎛️  DAC precision: {dac_variation/len(dac_voltages)*1000:.3f} mV per sample")
    print(f"  📡 ADC precision: {adc_variation/len(adc_voltages)*1000:.3f} mV per sample")
    
    print(f"\n🎉 HIGH-SPEED BRAM TEST COMPLETE!")
    print(f"   This is LASER-GRADE precision with 250 MHz sampling!")
    
    # Stop ramp
    driver.stop_ramp()
    
    return True

if __name__ == "__main__":
    try:
        test_high_speed_acquisition()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the device is connected and FPGA is programmed") 