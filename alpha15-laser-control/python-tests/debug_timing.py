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
    def get_decimation_rate(self):
        return self.client.recv_uint32()

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
    print("🔧 Alpha15 Timing and DMA Debug Tool")
    print("=" * 50)
    
    try:
        host = os.environ.get('HOST', '192.168.1.115')
        client = connect(host, 'alpha15-laser-control', restart=False)
        driver = CurrentRamp(client)
        print(f"✅ Connected to {host}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return 1

    # Test 1: Check basic ADC readings
    print("\n1. Testing basic ADC readings...")
    try:
        driver.select_adc_channel(0)
        driver.set_adc_input_range(0)  # 2 Vpp range
        
        for i in range(5):
            adc0 = driver.get_adc0_voltage()
            adc1 = driver.get_adc1_voltage()
            print(f"   Reading {i+1}: ADC0={adc0:.3f}V, ADC1={adc1:.3f}V")
            time.sleep(0.1)
    except Exception as e:
        print(f"   ❌ Basic ADC test failed: {e}")

    # Test 2: Check sample rate calculations
    print("\n2. Testing sample rate calculations...")
    try:
        test_rates = [100, 500, 1000, 2400, 4800]
        for rate in test_rates:
            driver.set_decimation_rate(rate)
            time.sleep(0.1)
            actual_rate = driver.get_decimated_sample_rate()
            expected_rate = 240_000_000 / rate
            print(f"   Decimation {rate}: Expected={expected_rate:.0f}Hz, Actual={actual_rate:.0f}Hz")
            if abs(actual_rate - expected_rate) > 1000:
                print(f"   ⚠️  Rate mismatch! Expected {expected_rate:.0f}, got {actual_rate:.0f}")
    except Exception as e:
        print(f"   ❌ Sample rate test failed: {e}")

    # Test 3: DMA streaming test
    print("\n3. Testing DMA streaming...")
    try:
        driver.set_decimation_rate(2400)  # 100 kHz
        expected_rate = 240_000_000 / 2400
        print(f"   Set decimation to 2400 (expected {expected_rate:.0f} Hz)")
        
        # Stop any existing streaming
        driver.stop_adc_streaming()
        time.sleep(0.5)
        
        # Start streaming
        driver.start_adc_streaming()
        time.sleep(1.0)  # Let it warm up
        
        # Check DMA status
        streaming = driver.is_adc_streaming_active()
        dma_running = driver.get_dma_running()
        dma_idle = driver.get_dma_idle()
        dma_error = driver.get_dma_error()
        
        print(f"   Streaming active: {streaming}")
        print(f"   DMA running: {dma_running}")
        print(f"   DMA idle: {dma_idle}")
        print(f"   DMA error: {dma_error}")
        
        if not dma_running:
            print("   ⚠️  DMA not running - this is the main problem!")
        
        # Monitor descriptor advancement
        print("   Monitoring descriptor advancement...")
        start_time = time.time()
        initial_desc = driver.get_current_descriptor_index()
        initial_pos = driver.get_buffer_position()
        
        time.sleep(2.0)  # Wait 2 seconds
        
        final_desc = driver.get_current_descriptor_index()
        final_pos = driver.get_buffer_position()
        elapsed = time.time() - start_time
        
        desc_change = (final_desc - initial_desc) % 512
        pos_change = (final_pos - initial_pos) % (512 * 2048)
        
        print(f"   Initial: desc={initial_desc}, pos={initial_pos}")
        print(f"   Final: desc={final_desc}, pos={final_pos}")
        print(f"   Change: desc={desc_change}, pos={pos_change} in {elapsed:.1f}s")
        
        if pos_change > 0:
            actual_sample_rate = pos_change / elapsed
            print(f"   Measured sample rate: {actual_sample_rate:.0f} Hz")
            if abs(actual_sample_rate - expected_rate) > 5000:
                print(f"   ⚠️  Sample rate mismatch! Expected {expected_rate:.0f}, measured {actual_sample_rate:.0f}")
        else:
            print("   ❌ No data advancement detected!")
        
    except Exception as e:
        print(f"   ❌ DMA streaming test failed: {e}")

    # Test 4: Data acquisition timing
    print("\n4. Testing data acquisition timing...")
    try:
        if driver.is_adc_streaming_active():
            print("   Testing 1-second data acquisition...")
            start_time = time.time()
            
            # Try to get 1 second worth of data
            samples_requested = int(expected_rate)  # 1 second worth
            data = driver.get_adc_stream_voltages(samples_requested)
            
            actual_time = time.time() - start_time
            
            print(f"   Requested {samples_requested} samples")
            print(f"   Received {len(data)} samples")
            print(f"   Time taken: {actual_time:.3f}s")
            
            if len(data) > 0:
                expected_time = len(data) / expected_rate
                print(f"   Expected time for {len(data)} samples: {expected_time:.3f}s")
                
                if actual_time < expected_time / 2:
                    print(f"   ⚠️  Data acquired too quickly! This suggests timing calibration issue.")
                elif actual_time > expected_time * 2:
                    print(f"   ⚠️  Data acquired too slowly! This suggests performance issue.")
                else:
                    print(f"   ✅ Timing appears correct")
                    
                # Check data quality
                data_min, data_max = np.min(data), np.max(data)
                data_std = np.std(data)
                print(f"   Data range: {data_min:.3f}V to {data_max:.3f}V")
                print(f"   Data std: {data_std:.3f}V")
                
                if data_std < 0.001:
                    print("   ⚠️  Data appears flat - check signal input")
            else:
                print("   ❌ No data received")
        else:
            print("   ❌ Streaming not active")
            
    except Exception as e:
        print(f"   ❌ Data acquisition test failed: {e}")

    # Test 5: Multiple channel test
    print("\n5. Testing ADC channel selection...")
    try:
        if driver.is_adc_streaming_active():
            for channel in [0, 1]:
                driver.select_adc_channel(channel)
                time.sleep(0.5)  # Let it settle
                
                data = driver.get_adc_stream_voltages(1000)
                if len(data) > 0:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    print(f"   Channel {channel}: mean={mean_val:.3f}V, std={std_val:.3f}V")
                else:
                    print(f"   Channel {channel}: No data")
    except Exception as e:
        print(f"   ❌ Channel selection test failed: {e}")

    # Cleanup
    try:
        driver.stop_adc_streaming()
        print("\n✅ Streaming stopped")
    except:
        pass

    print("\n🏁 Debug complete!")
    return 0

if __name__ == "__main__":
    exit(main()) 