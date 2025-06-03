#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import os
from koheron import connect, command

class VoltageControl(object):
    def __init__(self, client):
        self.client = client

    @command()
    def set_voltage_output(self, voltage):
        pass

    @command()
    def enable_output(self, enable):
        pass

    @command()
    def toggle_output(self):
        pass

    @command()
    def set_test_voltage(self):
        pass

    @command()
    def disable_test_voltage(self):
        pass

    @command()
    def get_output_voltage(self):
        return self.client.recv_float()

    @command()
    def is_output_enabled(self):
        return self.client.recv_bool()

# Test script for voltage control functionality
host = os.getenv('HOST', '192.168.1.20')  # Your Alpha250 IP
client = connect(host, 'vouttest', restart=True)
voltage_control = VoltageControl(client)

print("=== Alpha250 Voltage Control Test ===")
print(f"Connected to {host}")

try:
    # Test basic functionality
    print("\n1. Testing voltage setting...")
    voltage_control.set_voltage_output(0.5)
    print("   Set voltage to 0.5V")
    
    print("\n2. Testing output enable...")
    voltage_control.enable_output(True)
    print("   Output enabled")
    
    # Read back the status
    voltage = voltage_control.get_output_voltage()
    enabled = voltage_control.is_output_enabled()
    print(f"   Current voltage: {voltage:.3f}V")
    print(f"   Output enabled: {enabled}")
    
    print("\n3. Testing quick voltage function...")
    voltage_control.set_test_voltage()
    time.sleep(0.1)
    voltage = voltage_control.get_output_voltage()
    enabled = voltage_control.is_output_enabled()
    print(f"   Test voltage set - Voltage: {voltage:.3f}V, Enabled: {enabled}")
    
    print("\n4. Testing different voltages...")
    test_voltages = [1.0, 1.5, 2.0, 0.8]
    for v in test_voltages:
        voltage_control.set_voltage_output(v)
        time.sleep(0.1)
        actual_v = voltage_control.get_output_voltage()
        print(f"   Set {v}V -> Read {actual_v:.3f}V")
    
    print("\n5. Testing disable...")
    voltage_control.disable_test_voltage()
    enabled = voltage_control.is_output_enabled()
    print(f"   Output disabled: {not enabled}")
    
    print("\n✅ All tests completed successfully!")
    print("\nYou can now:")
    print(f"   - Open web interface at http://{host}")
    print("   - Use the toggle switch to enable/disable 0.5V output")
    print("   - Check Precision DAC Channel 0 with a multimeter")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    print("Make sure the bitstream is loaded and the server is running")

finally:
    try:
        # Safety: disable output before exit
        voltage_control.disable_test_voltage()
        print("\nSafety: Output disabled before exit")
    except:
        pass 