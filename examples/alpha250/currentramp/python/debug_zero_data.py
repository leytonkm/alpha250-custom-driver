#!/usr/bin/env python3

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
    def get_adc_samples_10000(self):
        return self.client.recv_array(10000, dtype='uint32')

    @command()
    def get_dac_samples_10000(self):
        return self.client.recv_array(10000, dtype='uint32')

    @command()
    def get_acquisition_count(self):
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

def debug_zero_data_issue():
    """
    Comprehensive diagnosis of why BRAM is capturing all zeros
    """
    
    # Connect to device
    host = os.environ.get('HOST', '192.168.1.20')
    print(f"🔍 Connecting to {host}...")
    
    try:
        client = connect(host, name='currentramp')
        driver = CurrentRamp(client)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return

    print("\n" + "="*60)
    print("🔍 ZERO DATA DIAGNOSTIC - COMPREHENSIVE ANALYSIS")
    print("="*60)

    # Step 1: Check if ramp is actually running
    print("\n1️⃣ CHECKING RAMP GENERATOR STATUS")
    print("-" * 40)
    
    try:
        ramp_enabled = driver.get_ramp_enabled()
        print(f"Ramp Enabled: {ramp_enabled}")
        
        if not ramp_enabled:
            print("⚠️  ISSUE FOUND: Ramp is not enabled!")
            print("   Starting ramp with test parameters...")
            
            # Configure and start ramp
            driver.set_ramp_frequency(10.0)  # 10 Hz
            driver.set_ramp_amplitude(1.0)   # 1V amplitude
            driver.set_ramp_offset(1.5)      # 1.5V offset
            driver.generate_ramp_waveform()
            driver.start_ramp()
            
            time.sleep(0.5)  # Wait for ramp to start
            
            ramp_enabled = driver.get_ramp_enabled()
            print(f"Ramp Enabled after start: {ramp_enabled}")
        
        # Check ramp phase accumulator
        phase = driver.get_ramp_phase()
        print(f"Ramp Phase: 0x{phase:08X} ({phase})")
        
        cycle_count = driver.get_cycle_count()
        print(f"Cycle Count: {cycle_count}")
        
        position = driver.get_ramp_position()
        print(f"Ramp Position: {position:.6f}")
        
        # Monitor phase changes over time
        print("\nMonitoring phase accumulator for 2 seconds...")
        phases = []
        for i in range(10):
            phase = driver.get_ramp_phase()
            phases.append(phase)
            print(f"  Phase[{i}]: 0x{phase:08X}")
            time.sleep(0.2)
        
        if len(set(phases)) == 1:
            print("❌ ISSUE: Phase accumulator is not changing - DDS not running!")
        else:
            print("✅ Phase accumulator is changing - DDS is running")
            
    except Exception as e:
        print(f"❌ Error checking ramp status: {e}")

    # Step 2: Check BRAM acquisition counter
    print("\n2️⃣ CHECKING BRAM ACQUISITION SYSTEM")
    print("-" * 40)
    
    try:
        # Monitor acquisition counter
        print("Monitoring BRAM acquisition counter...")
        counters = []
        for i in range(10):
            count = driver.get_acquisition_count()
            counters.append(count)
            print(f"  Acquisition Count[{i}]: {count}")
            time.sleep(0.2)
        
        if len(set(counters)) == 1:
            print("❌ ISSUE: Acquisition counter is not changing - BRAM not recording!")
            print("   This indicates the decimation counter or address counter is not working")
        else:
            print("✅ Acquisition counter is changing - BRAM is recording")
            
    except Exception as e:
        print(f"❌ Error checking acquisition counter: {e}")

    # Step 3: Test with manual DAC output
    print("\n3️⃣ TESTING WITH MANUAL DAC OUTPUT")
    print("-" * 40)
    
    try:
        print("Setting manual test voltage on DAC channel 2...")
        
        # Stop ramp first
        driver.stop_ramp()
        time.sleep(0.2)
        
        # Set a known test voltage
        test_voltage = 2.0  # 2V test signal
        driver.set_test_voltage_channel_2(test_voltage)
        print(f"Set DAC channel 2 to {test_voltage}V")
        
        time.sleep(1.0)  # Wait for BRAM to capture
        
        # Get BRAM data
        print("Reading BRAM data after manual DAC setting...")
        dac_data = driver.get_dac_samples_1000()
        
        # Check if we see the test voltage
        unique_values = np.unique(dac_data)
        print(f"Unique DAC values in BRAM: {unique_values[:10]}...")  # Show first 10
        
        if len(unique_values) == 1 and unique_values[0] == 0:
            print("❌ ISSUE: BRAM still shows all zeros even with manual DAC output!")
            print("   This indicates BRAM is not connected to DAC data")
        else:
            print("✅ BRAM is capturing DAC data")
            
            # Convert and analyze
            dac_voltages = []
            for raw in dac_data[:100]:  # Check first 100 samples
                dac_ch2_raw = raw & 0xFFFF
                voltage = (dac_ch2_raw / 65535.0) * 2.5
                dac_voltages.append(voltage)
            
            avg_voltage = np.mean(dac_voltages)
            print(f"Average DAC voltage from BRAM: {avg_voltage:.3f}V (expected: {test_voltage}V)")
            
    except Exception as e:
        print(f"❌ Error testing manual DAC: {e}")

    # Step 4: Check ADC with known input
    print("\n4️⃣ TESTING ADC INPUT")
    print("-" * 40)
    
    try:
        print("Reading ADC data from BRAM...")
        adc_data = driver.get_adc_samples_1000()
        
        unique_adc_values = np.unique(adc_data)
        print(f"Unique ADC values in BRAM: {unique_adc_values[:10]}...")
        
        if len(unique_adc_values) == 1 and unique_adc_values[0] == 0:
            print("❌ ISSUE: ADC BRAM shows all zeros!")
            print("   Possible causes:")
            print("   - ADC not connected to BRAM")
            print("   - ADC data path issue")
            print("   - BRAM address/enable signals wrong")
        else:
            print("✅ BRAM is capturing ADC data")
            
            # Convert and analyze
            adc_voltages = []
            for raw in adc_data[:100]:
                adc_ch0_raw = raw & 0xFFFF
                adc_signed = adc_ch0_raw if adc_ch0_raw < 32768 else adc_ch0_raw - 65536
                voltage = (adc_signed / 32768.0) * 1.8
                adc_voltages.append(voltage)
            
            avg_adc = np.mean(adc_voltages)
            std_adc = np.std(adc_voltages)
            print(f"ADC statistics: avg={avg_adc:.3f}V, std={std_adc:.3f}V")
            
    except Exception as e:
        print(f"❌ Error testing ADC: {e}")

    # Step 5: Hardware connectivity test
    print("\n5️⃣ HARDWARE CONNECTIVITY DIAGNOSIS")
    print("----------------------------------------")
    print("Testing hardware signal path...")
    
    # Ensure ramp is enabled with proper parameters
    print("🔧 Ensuring ramp is enabled with test parameters...")
    driver.set_ramp_frequency(10.0)  # 10Hz
    driver.set_ramp_amplitude(1.0)   # Full scale
    driver.set_ramp_offset(1.5)      # 1.5V center
    driver.generate_ramp_waveform()
    driver.start_ramp()
    
    # Wait for ramp to stabilize
    time.sleep(0.5)
    
    # Verify ramp is actually enabled
    ramp_enabled = driver.get_ramp_enabled()
    print(f"Ramp enabled before acquisition: {ramp_enabled}")
    
    if not ramp_enabled:
        print("❌ ERROR: Ramp is not enabled! Cannot capture ramp data.")
        return
    
    # CRITICAL: Trigger acquisition to enable BRAM write operations
    print("🔧 Triggering BRAM acquisition...")
    driver.trigger_acquisition()
    time.sleep(0.1)  # Allow time for data to be written
    
    print("Getting fresh BRAM data after acquisition trigger...")
    adc_data = driver.get_adc_samples_1000()
    dac_data = driver.get_dac_samples_1000()
    
    print(f"Raw ADC data - first 10 values: {adc_data[:10]}")
    print(f"Raw DAC data - first 10 values: {dac_data[:10]}")
    print(f"ADC data type: {type(adc_data[0])}, DAC data type: {type(dac_data[0])}")
    print(f"ADC data range: {np.min(adc_data)} to {np.max(adc_data)}")
    print(f"DAC data range: {np.min(dac_data)} to {np.max(dac_data)}")
    
    # Check if data is actually being written to different addresses
    adc_data_large = driver.get_adc_samples_10000()
    dac_data_large = driver.get_dac_samples_10000()
    print(f"Large ADC sample: min={np.min(adc_data_large)}, max={np.max(adc_data_large)}, unique={len(np.unique(adc_data_large))}")
    print(f"Large DAC sample: min={np.min(dac_data_large)}, max={np.max(dac_data_large)}, unique={len(np.unique(dac_data_large))}")
    
    # Check for any patterns in the data
    if len(np.unique(adc_data_large)) > 1:
        print(f"✅ ADC data has variation: {np.unique(adc_data_large)[:10]}")
    if len(np.unique(dac_data_large)) > 1:
        print(f"✅ DAC data has variation: {np.unique(dac_data_large)[:10]}")
    
    # Check for any non-zero values
    adc_nonzero = np.count_nonzero(adc_data)
    dac_nonzero = np.count_nonzero(dac_data)
    
    print(f"Non-zero ADC samples: {adc_nonzero}/1000")
    print(f"Non-zero DAC samples: {dac_nonzero}/1000")
    
    if adc_nonzero == 0 and dac_nonzero == 0:
        print("\n❌ CRITICAL ISSUE IDENTIFIED:")
        print("   Both ADC and DAC BRAM data are all zeros")
        print("   This indicates a fundamental hardware connection problem")
        print("\n🔧 LIKELY CAUSES:")
        print("   1. BRAM write enable (web) signal not connected")
        print("   2. BRAM address counter not working")
        print("   3. Decimation counter not generating enable pulses")
        print("   4. ADC/DAC data not reaching BRAM inputs")
        print("   5. Clock domain issues")
        
    elif dac_nonzero > 0 and adc_nonzero == 0:
        print("\n⚠️  PARTIAL ISSUE:")
        print("   DAC data is being captured but ADC data is all zeros")
        print("   ADC connection to BRAM may be the issue")
        
    elif adc_nonzero > 0 and dac_nonzero == 0:
        print("\n⚠️  PARTIAL ISSUE:")
        print("   ADC data is being captured but DAC data is all zeros")
        print("   DAC connection to BRAM may be the issue")
        
    else:
        print("\n✅ Both ADC and DAC have non-zero data - investigating values...")

    print("\n" + "="*60)
    print("🎯 DIAGNOSTIC SUMMARY")
    print("="*60)
    print("Based on the web interface logs showing all zeros,")
    print("the most likely issue is in the FPGA hardware design:")
    print()
    print("1. Check block_design.tcl BRAM connections")
    print("2. Verify decimation counter is generating pulses")
    print("3. Confirm address counter is incrementing")
    print("4. Validate ADC/DAC data paths to BRAM")
    print("5. Check clock domain crossing")
    print()
    print("Run this script to get detailed hardware diagnostics.")

if __name__ == "__main__":
    debug_zero_data_issue() 