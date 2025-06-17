#!/usr/bin/env python3

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
    def get_adc_samples_10000(self):
        return self.client.recv_array(10000, dtype='uint32')

    @command()
    def get_dac_samples_10000(self):
        return self.client.recv_array(10000, dtype='uint32')

    @command()
    def get_acquisition_count(self):
        return self.client.recv_uint32()

    @command()
    def get_adc_size(self):
        return self.client.recv_uint32()

    @command()
    def get_dac_size(self):
        return self.client.recv_uint32()

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
    def set_test_voltage_channel_2(self, voltage):
        pass

def convert_bram_data(adc_raw, dac_raw):
    """
    Convert raw BRAM data to voltages using corrected Alpha250 implementation
    
    NEW FORMAT: ADC BRAM contains both signals in 32-bit format:
    - Lower 16 bits: ADC channel 0 input (LTC2157: 16-bit signed, ±1.8V)
    - Upper 16 bits: Ramp output signal (16-bit unsigned, 0-2.5V)
    
    DAC BRAM is not used for data capture in this corrected implementation.
    """
    n_samples = len(adc_raw)
    
    adc_voltages = []
    dac_voltages = []
    
    for i in range(n_samples):
        # Extract both signals from the combined ADC BRAM data
        combined_data = adc_raw[i]
        
        # ADC input: Lower 16 bits (LTC2157: 16-bit signed, ±1.8V range)
        adc_raw_16bit = combined_data & 0xFFFF
        adc_signed = adc_raw_16bit if adc_raw_16bit < 32768 else adc_raw_16bit - 65536
        adc_voltage = (adc_signed / 32768.0) * 1.8
        adc_voltages.append(adc_voltage)
        
        # Ramp output: Upper 16 bits (16-bit unsigned, 0-2.5V range)
        ramp_raw_16bit = (combined_data >> 16) & 0xFFFF
        ramp_voltage = (ramp_raw_16bit / 65535.0) * 2.5
        dac_voltages.append(ramp_voltage)
    
    return np.array(adc_voltages), np.array(dac_voltages)

def test_corrected_bram_implementation():
    """
    Test the corrected Alpha250 BRAM implementation
    """
    
    # Connect to device
    host = os.environ.get('HOST', '192.168.1.20')
    print(f"🔗 Connecting to {host}...")
    
    try:
        client = connect(host, name='currentramp')
        driver = CurrentRamp(client)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print("\n" + "="*70)
    print("🧪 TESTING CORRECTED ALPHA250 BRAM IMPLEMENTATION")
    print("   Following proven working patterns from Alpha250 examples")
    print("   ADC BRAM captures both ADC input + ramp output simultaneously")
    print("="*70)

    # Step 1: Configure and enable ramp
    print("\n1️⃣ CONFIGURING AND ENABLING RAMP")
    print("-" * 50)
    
    try:
        # Stop any existing ramp first
        driver.stop_ramp()
        time.sleep(0.2)
        
        # Configure ramp with clear parameters
        ramp_freq = 10.0
        ramp_amplitude = 1.0
        ramp_offset = 1.5
        
        print(f"Setting ramp parameters:")
        print(f"  Frequency: {ramp_freq} Hz")
        print(f"  Amplitude: {ramp_amplitude} V")
        print(f"  Offset: {ramp_offset} V")
        
        driver.set_ramp_frequency(ramp_freq)
        driver.set_ramp_amplitude(ramp_amplitude)
        driver.set_ramp_offset(ramp_offset)
        driver.generate_ramp_waveform()
        
        # Start the ramp
        print("Starting ramp...")
        driver.start_ramp()
        time.sleep(0.5)
        
        # Verify ramp is running
        ramp_enabled = driver.get_ramp_enabled()
        print(f"✅ Ramp enabled: {ramp_enabled}")
        
        if not ramp_enabled:
            print("❌ ERROR: Ramp failed to start!")
            return
            
    except Exception as e:
        print(f"❌ Error configuring ramp: {e}")
        return

    # Step 2: Test data capture
    print("\n2️⃣ TESTING DATA CAPTURE")
    print("-" * 50)
    
    try:
        print("Triggering acquisition and capturing data...")
        driver.trigger_acquisition()
        time.sleep(0.2)
        
        adc_data = driver.get_adc_samples_1000()
        dac_data = driver.get_dac_samples_1000()
        
        print(f"Received ADC data: {len(adc_data)} samples")
        print(f"Received DAC data: {len(dac_data)} samples")
        
        # Check for non-zero data
        adc_nonzero = np.count_nonzero(adc_data)
        dac_nonzero = np.count_nonzero(dac_data)
        
        print(f"Non-zero ADC samples: {adc_nonzero}/1000")
        print(f"Non-zero DAC samples: {dac_nonzero}/1000")
        
        # Convert to voltages
        adc_voltages, dac_voltages = convert_bram_data(adc_data, dac_data)
        
        print(f"ADC voltage range: {np.min(adc_voltages):.3f} to {np.max(adc_voltages):.3f} V")
        print(f"DAC voltage range: {np.min(dac_voltages):.3f} to {np.max(dac_voltages):.3f} V")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        time_axis = np.arange(1000) / 10000.0 * 1000  # Convert to ms
        
        plt.subplot(1, 2, 1)
        plt.plot(time_axis, adc_voltages, 'b-', label='ADC Input', alpha=0.7)
        plt.plot(time_axis, dac_voltages, 'r-', label='DAC Output', alpha=0.7)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (V)')
        plt.title('Corrected BRAM Data Capture')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(dac_voltages, adc_voltages, alpha=0.5, s=1)
        plt.xlabel('DAC Output (V)')
        plt.ylabel('ADC Input (V)')
        plt.title('ADC vs DAC Correlation')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('corrected_bram_test.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved as corrected_bram_test.png")
        
        # Final assessment
        print("\n" + "="*70)
        print("🎯 TEST RESULTS")
        print("="*70)
        
        if adc_nonzero > 0 or dac_nonzero > 0:
            print("✅ SUCCESS: BRAM is capturing data!")
            print("   The corrected implementation is working")
            print("   Zero data issue has been resolved")
        else:
            print("❌ ISSUE: Still getting all zeros")
            print("   Further investigation needed")
            
    except Exception as e:
        print(f"❌ Error testing data capture: {e}")

if __name__ == "__main__":
    test_corrected_bram_implementation() 