#!/usr/bin/env python3

import os
import time
import numpy as np
from koheron import connect, command

class CurrentRamp:
    def __init__(self, client):
        self.client = client

    @command()
    def get_photodiode_voltage(self):
        return self.client.recv_float()

    @command()
    def get_photodiode_raw(self):
        return self.client.recv_uint32()

    @command()
    def set_photodiode_channel(self, channel):
        pass

def test_simple_adc():
    """
    Simple test to just read ADC values directly
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

    print("\n" + "="*50)
    print("🧪 SIMPLE ADC TEST")
    print("="*50)

    # Test both ADC channels
    for channel in [0, 1]:
        print(f"\n📊 Testing ADC Channel {channel}")
        print("-" * 30)
        
        try:
            # Set the channel
            driver.set_photodiode_channel(channel)
            time.sleep(0.1)
            
            # Read multiple samples
            voltages = []
            raw_values = []
            
            for i in range(10):
                voltage = driver.get_photodiode_voltage()
                raw = driver.get_photodiode_raw()
                
                voltages.append(voltage)
                raw_values.append(raw)
                
                print(f"  Sample {i+1}: {voltage:.6f}V (raw: {raw})")
                time.sleep(0.1)
            
            # Statistics
            avg_voltage = np.mean(voltages)
            std_voltage = np.std(voltages)
            min_voltage = np.min(voltages)
            max_voltage = np.max(voltages)
            
            print(f"\n📈 Channel {channel} Statistics:")
            print(f"  Average: {avg_voltage:.6f}V")
            print(f"  Std Dev: {std_voltage:.6f}V")
            print(f"  Range: {min_voltage:.6f}V to {max_voltage:.6f}V")
            print(f"  Raw range: {np.min(raw_values)} to {np.max(raw_values)}")
            
            if std_voltage > 0.001:
                print(f"  ✅ Signal has variation ({std_voltage:.6f}V)")
            else:
                print(f"  ⚠️  Signal appears static ({std_voltage:.6f}V)")
                
        except Exception as e:
            print(f"❌ Error reading channel {channel}: {e}")

    print("\n" + "="*50)
    print("🎯 SIMPLE ADC TEST COMPLETE")
    print("="*50)
    print("If you see voltage readings, the ADC is working!")
    print("If you see your -0.5 to 0.5V sine wave, connect it to the ADC input.")

if __name__ == "__main__":
    test_simple_adc() 