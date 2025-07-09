#!/usr/bin/env python3

import os
import time
import numpy as np
from koheron import connect, command

class CurrentRamp:
    """Debug interface for CurrentRamp driver"""
    def __init__(self, client):
        self.client = client

    @command('CurrentRamp')
    def select_adc_channel(self, channel):
        pass

    @command('CurrentRamp')
    def set_decimation_rate(self, rate):
        pass

    @command('CurrentRamp')
    def get_decimated_sample_rate(self):
        return self.client.recv_double()

    @command('CurrentRamp')
    def start_adc_streaming(self):
        pass

    @command('CurrentRamp')
    def stop_adc_streaming(self):
        pass

    @command('CurrentRamp')
    def is_adc_streaming_active(self):
        return self.client.recv_bool()

    @command('CurrentRamp')
    def get_dma_running(self):
        return self.client.recv_bool()

    @command('CurrentRamp')
    def get_dma_idle(self):
        return self.client.recv_bool()

    @command('CurrentRamp')
    def get_dma_error(self):
        return self.client.recv_bool()

    @command('CurrentRamp')
    def get_current_descriptor_index(self):
        return self.client.recv_uint32()

    @command('CurrentRamp')
    def get_buffer_position(self):
        return self.client.recv_uint32()

    @command('CurrentRamp')
    def get_adc_stream_voltages(self, num_samples):
        return self.client.recv_vector(dtype='float32')

    @command('CurrentRamp')
    def get_adc0_voltage(self):
        return self.client.recv_float()

    @command('CurrentRamp')
    def get_adc1_voltage(self):
        return self.client.recv_float()

    @command('CurrentRamp')
    def set_adc_input_range(self, range_sel):
        pass

def main():
    print("🔧 Alpha15 ADC Debug Tool")
    print("=" * 50)
    
    try:
        host = os.environ.get('HOST', '192.168.1.115')
        client = connect(host, 'alpha15-laser-control', restart=False)
        driver = CurrentRamp(client)
        print(f"✅ Connected to {host}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    print("\n1. Testing basic ADC readings...")
    try:
        adc0 = driver.get_adc0_voltage()
        adc1 = driver.get_adc1_voltage()
        print(f"   ADC0: {adc0:.3f} V")
        print(f"   ADC1: {adc1:.3f} V")
    except Exception as e:
        print(f"   ❌ Basic ADC read failed: {e}")

    print("\n2. Setting up ADC streaming...")
    try:
        # Select channel 0 (RFADC0)
        driver.select_adc_channel(0)
        print("   ✅ Selected ADC channel 0")
        
        # Set ADC range to 2 Vpp (more sensitive)
        driver.set_adc_input_range(0)
        print("   ✅ Set ADC range to 2 Vpp")
        
        # Set reasonable decimation rate
        driver.set_decimation_rate(2400)  # 240MHz/2400 = 100kHz
        sample_rate = driver.get_decimated_sample_rate()
        print(f"   ✅ Decimation rate set, sample rate: {sample_rate/1000:.1f} kHz")
        
    except Exception as e:
        print(f"   ❌ Setup failed: {e}")
        return 1

    print("\n3. Starting DMA streaming...")
    try:
        driver.start_adc_streaming()
        time.sleep(0.5)  # Let it warm up
        
        streaming = driver.is_adc_streaming_active()
        dma_running = driver.get_dma_running()
        dma_idle = driver.get_dma_idle()
        dma_error = driver.get_dma_error()
        
        print(f"   Streaming active: {streaming}")
        print(f"   DMA running: {dma_running}")
        print(f"   DMA idle: {dma_idle}")
        print(f"   DMA error: {dma_error}")
        
        if not dma_running:
            print("   ⚠️  DMA not running - this is the problem!")
        
    except Exception as e:
        print(f"   ❌ Streaming start failed: {e}")
        return 1

    print("\n4. Testing data acquisition...")
    try:
        for i in range(5):
            time.sleep(0.2)
            desc_idx = driver.get_current_descriptor_index()
            buf_pos = driver.get_buffer_position()
            print(f"   Iteration {i+1}: desc={desc_idx}, buf_pos={buf_pos}")
            
            # Try to get some data
            try:
                data = driver.get_adc_stream_voltages(1000)
                if len(data) > 0:
                    print(f"   ✅ Got {len(data)} samples, range: {np.min(data):.3f} to {np.max(data):.3f} V")
                    print(f"   Mean: {np.mean(data):.3f} V, Std: {np.std(data):.3f} V")
                else:
                    print("   ❌ No data received")
            except Exception as e:
                print(f"   ❌ Data read failed: {e}")
    
    except Exception as e:
        print(f"   ❌ Data acquisition test failed: {e}")

    print("\n5. Stopping streaming...")
    try:
        driver.stop_adc_streaming()
        print("   ✅ Streaming stopped")
    except Exception as e:
        print(f"   ❌ Stop failed: {e}")

    print("\n🏁 Debug complete!")
    return 0

if __name__ == "__main__":
    exit(main()) 