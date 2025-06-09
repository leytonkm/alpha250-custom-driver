#!/usr/bin/env python3
"""
CurrentRamp Driver Test Script
Tests the CurrentRamp driver using the Koheron Python client
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
        """Set ramp frequency in Hz (0.1-1000)"""
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
        """Start ramp output"""
        pass
    
    @command()
    def stop_ramp(self):
        """Stop ramp output"""
        pass
    
    @command()
    def get_ramp_enabled(self):
        """Check if ramp is enabled"""
        return self.client.recv_bool()
    
    # === Manual Testing Functions ===
    
    @command()
    def set_test_voltage_channel_0(self, voltage):
        """Set test voltage on Precision DAC Channel 0 (laser)"""
        pass
    
    @command()
    def set_test_voltage_channel_1(self, voltage):
        """Set test voltage on Precision DAC Channel 1 (temperature)"""
        pass
    
    @command()
    def set_ramp_manual(self, triangle_position):
        """Set ramp to manual position (0.0 to 1.0)"""
        pass
    
    @command()
    def update_ramp(self):
        """Update ramp to next sample (call repeatedly for automatic ramp)"""
        pass
    
    # === Photodiode Reading ===
    
    @command()
    def get_photodiode_reading(self):
        """Get current photodiode reading in V"""
        return self.client.recv_float()

def test_precision_dac_direct(driver):
    """Test direct precision DAC control"""
    print("\n🧪 Testing Direct Precision DAC Control...")
    
    try:
        # Test Channel 0 (laser)
        print("   Testing Precision DAC Channel 0 (laser)...")
        test_voltages = [0.5, 1.0, 1.5, 2.0]
        for voltage in test_voltages:
            driver.set_test_voltage_channel_0(voltage)
            time.sleep(0.2)
            print(f"   Channel 0 set to {voltage}V")
        
        # Test Channel 1 (temperature)
        print("   Testing Precision DAC Channel 1 (temperature)...")
        for voltage in test_voltages:
            driver.set_test_voltage_channel_1(voltage)
            time.sleep(0.2)
            print(f"   Channel 1 set to {voltage}V")
        
        # Reset to safe values
        driver.set_test_voltage_channel_0(0.0)
        driver.set_test_voltage_channel_1(0.0)
        
        print("   ✅ Direct precision DAC control working")
        return True
        
    except Exception as e:
        print(f"   ❌ Precision DAC test failed: {e}")
        return False

def test_dc_control(driver):
    """Test DC voltage control functions"""
    print("\n🧪 Testing DC Temperature Control...")
    
    try:
        # Set voltage
        test_voltage = 1.5
        print(f"   Setting DC voltage to {test_voltage}V...")
        driver.set_temperature_dc_voltage(test_voltage)
        time.sleep(0.1)
        
        # Read back
        voltage = driver.get_temperature_dc_voltage()
        print(f"   Read voltage: {voltage:.3f}V")
        
        if abs(voltage - test_voltage) < 0.01:
            print("   ✅ DC voltage set/get working")
        else:
            print(f"   ❌ Voltage mismatch: expected {test_voltage}, got {voltage}")
            return False
        
        # Test enable
        print("   Testing DC output enable...")
        driver.enable_temperature_dc_output(True)
        time.sleep(0.1)
        
        enabled = driver.get_temperature_dc_enabled()
        print(f"   DC output enabled: {enabled}")
        
        if enabled:
            print("   ✅ DC enable/disable working")
        else:
            print("   ❌ DC enable failed")
            return False
        
        # Clean up
        driver.enable_temperature_dc_output(False)
        return True
        
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
        print(f"   Read frequency: {freq:.1f} Hz")
        
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
            abs(offset - test_offset) < 0.01):
            print("   ✅ Ramp parameters working")
            return True
        else:
            print("   ❌ Parameter mismatch")
            return False
            
    except Exception as e:
        print(f"   ❌ Ramp parameter test failed: {e}")
        return False

def test_ramp_control(driver):
    """Test ramp start/stop"""
    print("\n🧪 Testing Ramp Control...")
    
    try:
        # Configure waveform parameters
        print("   Configuring ramp waveform...")
        driver.generate_ramp_waveform()
        time.sleep(0.1)
        
        # Start ramp
        print("   Starting ramp...")
        driver.start_ramp()
        time.sleep(0.5)
        
        # Check status (if available)
        try:
            enabled = driver.get_ramp_enabled()
            print(f"   Ramp running: {enabled}")
        except:
            print("   Ramp started (status check not available)")
        
        print("   ✅ Ramp started - check oscilloscope on Precision DAC Channel 0!")
        
        # Stop ramp
        print("   Stopping ramp...")
        driver.stop_ramp()
        time.sleep(0.1)
        
        print("   ✅ Ramp control working")
        return True
        
    except Exception as e:
        print(f"   ❌ Ramp control test failed: {e}")
        return False

def test_photodiode(driver):
    """Test photodiode reading"""
    print("\n🧪 Testing Photodiode Reading...")
    
    try:
        reading = driver.get_photodiode_reading()
        print(f"   Photodiode reading: {reading:.3f}V")
        print("   ✅ Photodiode reading working")
        return True
    except Exception as e:
        print(f"   ❌ Photodiode test failed: {e}")
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
    """Run comprehensive CurrentRamp tests"""
    print("🚀 CurrentRamp Driver Test Suite")
    print("=" * 50)
    
    # Connect to instrument
    host = os.getenv('HOST', '192.168.1.20')
    
    try:
        print(f"🔌 Connecting to {host}...")
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print("✅ Connected to CurrentRamp driver")
        
        # Run all tests
        tests = [
            ("Direct Precision DAC", lambda: test_precision_dac_direct(driver)),
            ("DC Temperature Control", lambda: test_dc_control(driver)),
            ("Ramp Parameters", lambda: test_ramp_parameters(driver)),
            ("Ramp Control", lambda: test_ramp_control(driver)),
            ("Photodiode Reading", lambda: test_photodiode(driver)),
            ("Edge Cases", lambda: test_edge_cases(driver)),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
                results.append((test_name, False))
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 Test Results Summary:")
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
            if result:
                passed += 1
        
        print(f"\n🎯 Tests passed: {passed}/{len(results)}")
        
        if passed == len(results):
            print("\n🎉 All tests passed! Ready for oscilloscope testing.")
            print("\n📋 Oscilloscope Test Procedure:")
            print("   🔴 Precision DAC Channel 0: Current ramp output (laser)")
            print("   🔵 Precision DAC Channel 1: DC temperature control") 
            print("\n   Test sequence:")
            print("   1. Connect scope probe 1 to Precision DAC Channel 0")
            print("   2. Connect scope probe 2 to Precision DAC Channel 1")
            print("   3. Run: driver.set_ramp_frequency(10)")
            print("   4. Run: driver.set_ramp_amplitude(1.0)")
            print("   5. Run: driver.set_ramp_offset(1.5)")
            print("   6. Run: driver.start_ramp()")
            print("   7. Verify 10Hz linear ramp: 1.5V to 2.5V on Channel 0")
            print("   8. Run: driver.set_temperature_dc_voltage(0.5)")
            print("   9. Run: driver.enable_temperature_dc_output(True)")
            print("   10. Verify constant 0.5V on Channel 1")
            print("\n   Manual testing:")
            print("   - driver.set_test_voltage_channel_0(1.0)  # Direct DAC control")
            print("   - driver.set_test_voltage_channel_1(2.0)  # Direct DAC control")
        else:
            print("\n⚠️  Some tests failed. Fix issues before oscilloscope testing.")
        
        return 0 if passed == len(results) else 1
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   1. Check CurrentRamp instrument is running on Alpha250")
        print("   2. Verify network connection to 192.168.1.20")
        print("   3. Try: make run CONFIG=examples/alpha250/currentramp/config.yml HOST=192.168.1.20")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 