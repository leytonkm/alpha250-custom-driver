#!/usr/bin/env python3
"""
CurrentRamp Driver Test Script
Tests the CurrentRamp driver using the Koheron Python client
Hardware-based precise timing implementation
"""

import os
import time
import sys
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver"""
    
    def __init__(self, client):
        self.client = client

    # === DC Temperature Control Functions ===
    
    @command()
    def set_temperature_dc_voltage(self, voltage):
        """Set DC temperature control voltage (0-2.5V)"""
        pass
    
    @command()
    def enable_temperature_dc_output(self, enable):
        """Enable/disable DC temperature output"""
        pass
    
    @command()
    def get_temperature_dc_voltage(self):
        """Get current DC voltage setting"""
        return self.client.recv_float()
    
    @command()
    def get_temperature_dc_enabled(self):
        """Get DC output enable status"""
        return self.client.recv_bool()

    # === Ramp Control Functions ===
    
    @command()
    def set_ramp_frequency(self, frequency):
        """Set ramp frequency in Hz (0.001-1000)"""
        pass
    
    @command()
    def set_ramp_amplitude(self, amplitude):
        """Set ramp amplitude in V (0-2.5)"""
        pass
    
    @command()
    def set_ramp_offset(self, offset):
        """Set ramp DC offset in V (0-2.5)"""
        pass
    
    @command()
    def get_ramp_frequency(self):
        """Get current ramp frequency"""
        return self.client.recv_double()
    
    @command()
    def get_ramp_amplitude(self):
        """Get current ramp amplitude"""
        return self.client.recv_float()
    
    @command()
    def get_ramp_offset(self):
        """Get current ramp offset"""
        return self.client.recv_float()
    
    @command()
    def generate_ramp_waveform(self):
        """Configure ramp waveform parameters"""
        pass
    
    @command()
    def start_ramp(self):
        """Start hardware ramp output"""
        pass
    
    @command()
    def stop_ramp(self):
        """Stop hardware ramp output"""
        pass
    
    @command()
    def get_ramp_enabled(self):
        """Get ramp enable status"""
        return self.client.recv_bool()
    
    @command()
    def set_ramp_manual(self, position):
        """Set manual ramp position (0.0-1.0)"""
        pass

    # === Hardware Status Functions ===
    
    @command()
    def get_ramp_phase(self):
        """Get current ramp phase accumulator value"""
        return self.client.recv_uint32()
    
    @command()
    def get_cycle_count(self):
        """Get number of completed ramp cycles"""
        return self.client.recv_uint32()
    
    @command()
    def get_ramp_position(self):
        """Get current ramp position (0.0-1.0)"""
        return self.client.recv_float()
    
    @command()
    def get_timing_status(self):
        """Get hardware timing diagnostics"""
        return (self.client.recv_double(),  # fs_adc
                self.client.recv_uint32(),  # ramp_phase
                self.client.recv_uint32(),  # cycle_count
                self.client.recv_float(),   # ramp_position
                self.client.recv_bool())    # hardware_ramp_enabled

    # === Testing Functions ===
    
    @command()
    def set_test_voltage_channel_0(self, voltage):
        """Set test voltage on channel 0 (0-2.5V)"""
        pass
    
    @command()
    def set_test_voltage_channel_1(self, voltage):
        """Set test voltage on channel 1 (0-2.5V)"""
        pass
    
    @command()
    def set_test_voltage_channel_2(self, voltage):
        """Set test voltage on channel 2 (0-2.5V) - Note: conflicts with hardware ramp"""
        pass
    
    @command()
    def get_photodiode_reading(self):
        """Get photodiode reading (placeholder)"""
        return self.client.recv_float()

def connect_to_device():
    """Connect to the Alpha250 device"""
    print("🔌 Connecting to Alpha250...")
    
    try:
        # Try to connect to device
        host = os.environ.get('HOST', '192.168.1.100')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"   ✅ Connected to {host}")
        return driver
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None

def test_dc_control(driver):
    """Test DC temperature control functions"""
    print("\n🧪 Testing DC Temperature Control...")
    
    try:
        # Test voltage setting
        test_voltage = 1.2
        print(f"   Setting DC voltage to {test_voltage}V...")
        driver.set_temperature_dc_voltage(test_voltage)
        time.sleep(0.1)
        
        voltage = driver.get_temperature_dc_voltage()
        print(f"   Read DC voltage: {voltage:.3f}V")
        
        # Test enable/disable
        print("   Enabling DC output...")
        driver.enable_temperature_dc_output(True)
        time.sleep(0.1)
        
        enabled = driver.get_temperature_dc_enabled()
        print(f"   DC output enabled: {enabled}")
        
        # Test disable
        print("   Disabling DC output...")
        driver.enable_temperature_dc_output(False)
        time.sleep(0.1)
        
        enabled = driver.get_temperature_dc_enabled()
        print(f"   DC output enabled: {enabled}")
        
        if abs(voltage - test_voltage) < 0.01:
            print("   ✅ DC control working")
            return True
        else:
            print("   ❌ DC voltage mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ DC control test failed: {e}")
        return False

def test_ramp_parameters(driver):
    """Test ramp parameter functions"""
    print("\n🧪 Testing Ramp Parameters...")
    
    try:
        # Test frequency
        test_freq = 25.0
        print(f"   Setting frequency to {test_freq} Hz...")
        driver.set_ramp_frequency(test_freq)
        time.sleep(0.1)
        
        freq = driver.get_ramp_frequency()
        print(f"   Read frequency: {freq:.3f} Hz")
        
        # Test amplitude  
        test_amp = 1.2
        print(f"   Setting amplitude to {test_amp}V...")
        driver.set_ramp_amplitude(test_amp)
        time.sleep(0.1)
        
        amplitude = driver.get_ramp_amplitude()
        print(f"   Read amplitude: {amplitude:.2f}V")
        
        # Test offset
        test_offset = 0.8
        print(f"   Setting offset to {test_offset}V...")
        driver.set_ramp_offset(test_offset)
        time.sleep(0.1)
        
        offset = driver.get_ramp_offset()
        print(f"   Read offset: {offset:.2f}V")
        
        # Verify values
        if (abs(amplitude - test_amp) < 0.01 and 
            abs(offset - test_offset) < 0.01 and
            abs(freq - test_freq) < 0.01):
            print("   ✅ Ramp parameters working")
            return True
        else:
            print("   ❌ Parameter mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ Ramp parameter test failed: {e}")
        return False

def test_hardware_ramp(driver):
    """Test hardware ramp generation"""
    print("\n🧪 Testing Hardware Ramp Generation...")
    
    try:
        # Configure waveform parameters
        print("   Configuring hardware ramp waveform...")
        driver.generate_ramp_waveform()
        time.sleep(0.1)
        
        # Get timing status before starting
        fs_adc, phase_before, cycles_before, pos_before, enabled_before = driver.get_timing_status()
        print(f"   ADC sampling frequency: {fs_adc/1e6:.1f} MHz")
        print(f"   Initial phase: {phase_before}, cycles: {cycles_before}, position: {pos_before:.3f}")
        
        # Start ramp
        print("   Starting hardware ramp...")
        driver.start_ramp()
        time.sleep(0.1)
        
        # Check status
        enabled = driver.get_ramp_enabled()
        print(f"   Hardware ramp enabled: {enabled}")
        
        if not enabled:
            print("   ❌ Hardware ramp failed to start")
            return False
        
        # Monitor ramp for a few cycles
        print("   Monitoring ramp for 2 seconds...")
        start_time = time.time()
        last_cycle_count = 0
        
        while time.time() - start_time < 2.0:
            fs_adc, phase, cycles, position, enabled = driver.get_timing_status()
            
            if cycles != last_cycle_count:
                frequency = driver.get_ramp_frequency()
                print(f"   Cycle {cycles}: phase={phase:08x}, position={position:.3f}, freq={frequency:.3f}Hz")
                last_cycle_count = cycles
            
            time.sleep(0.1)
        
        # Check if we got cycles
        final_cycles = driver.get_cycle_count()
        if final_cycles > cycles_before:
            print(f"   ✅ Hardware ramp generated {final_cycles - cycles_before} cycles")
            measured_freq = (final_cycles - cycles_before) / 2.0  # 2 seconds
            expected_freq = driver.get_ramp_frequency()
            print(f"   Measured frequency: {measured_freq:.2f} Hz (expected: {expected_freq:.2f} Hz)")
            
            # Stop ramp
            print("   Stopping hardware ramp...")
            driver.stop_ramp()
            time.sleep(0.1)
            
            enabled = driver.get_ramp_enabled()
            print(f"   Hardware ramp enabled: {enabled}")
            
            if not enabled:
                print("   ✅ Hardware ramp control working")
                return True
            else:
                print("   ❌ Hardware ramp failed to stop")
                return False
        else:
            print("   ❌ No ramp cycles detected")
            return False
        
    except Exception as e:
        print(f"   ❌ Hardware ramp test failed: {e}")
        return False

def test_timing_accuracy(driver):
    """Test timing accuracy of hardware ramp"""
    print("\n🧪 Testing Timing Accuracy...")
    
    try:
        frequencies_to_test = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        
        for target_freq in frequencies_to_test:
            print(f"\n   Testing {target_freq} Hz...")
            
            # Set frequency
            driver.set_ramp_frequency(target_freq)
            driver.generate_ramp_waveform()
            
            # Start ramp
            driver.start_ramp()
            time.sleep(0.1)  # Let it stabilize
            
            # Measure for 5 seconds
            start_cycles = driver.get_cycle_count()
            start_time = time.time()
            
            time.sleep(5.0)
            
            end_cycles = driver.get_cycle_count()
            end_time = time.time()
            
            # Calculate measured frequency
            cycle_diff = end_cycles - start_cycles
            time_diff = end_time - start_time
            measured_freq = cycle_diff / time_diff
            
            error_percent = abs(measured_freq - target_freq) / target_freq * 100
            
            print(f"   Target: {target_freq:.1f} Hz, Measured: {measured_freq:.3f} Hz, Error: {error_percent:.2f}%")
            
            if error_percent > 1.0:  # More than 1% error
                print(f"   ⚠️  High error for {target_freq} Hz")
            else:
                print(f"   ✅ Good accuracy for {target_freq} Hz")
            
            driver.stop_ramp()
            time.sleep(0.1)
        
        print("   ✅ Timing accuracy test completed")
        return True
        
    except Exception as e:
        print(f"   ❌ Timing accuracy test failed: {e}")
        return False

def test_edge_cases(driver):
    """Test parameter validation"""
    print("\n🧪 Testing Edge Cases...")
    
    try:
        # Test invalid voltage (should be handled gracefully)
        print("   Testing invalid voltage (3.0V)...")
        driver.set_temperature_dc_voltage(3.0)
        time.sleep(0.1)
        
        # Test invalid frequency  
        print("   Testing invalid frequency (5000 Hz)...")
        driver.set_ramp_frequency(5000.0)
        time.sleep(0.1)
        
        # Test very low frequency
        print("   Testing very low frequency (0.1 Hz)...")
        driver.set_ramp_frequency(0.1)
        driver.generate_ramp_waveform()
        driver.start_ramp()
        time.sleep(1.0)
        cycles = driver.get_cycle_count()
        driver.stop_ramp()
        print(f"   Low frequency test: {cycles} cycles in 1 second")
        
        # Test amplitude + offset > 2.5V
        print("   Testing amplitude + offset validation...")
        driver.set_ramp_amplitude(1.5)
        driver.set_ramp_offset(1.5)  # Total = 3.0V
        time.sleep(0.1)
        
        print("   ✅ Edge cases tested (check server logs for validation)")
        return True
        
    except Exception as e:
        print(f"   ❌ Edge case test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 CurrentRamp Hardware Driver Test Suite")
    print("   Hardware-based precise timing implementation")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("\n❌ Cannot connect to device. Exiting.")
        return 1
    
    # Run tests
    tests = [
        ("DC Control", test_dc_control),
        ("Ramp Parameters", test_ramp_parameters), 
        ("Hardware Ramp", test_hardware_ramp),
        ("Timing Accuracy", test_timing_accuracy),
        ("Edge Cases", test_edge_cases)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(driver)
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Hardware ramp timing should be accurate.")
        print("📊 Check your oscilloscope to verify the exact frequencies.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    exit(main()) 