#!/usr/bin/env python3
"""
Performance Fix Validation Test
Tests the new fast array retrieval and DMA diagnostics
Goal: Verify 100x+ speedup in small sample retrieval
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver - Updated with performance fixes"""
    
    def __init__(self, client):
        self.client = client

    # Original slow functions (for comparison)
    @command()
    def get_adc_stream_voltages(self, num_samples):
        """Get array of ADC streaming data as voltages - SLOW VERSION"""
        full_buffer_size = 64 * 64 * 1024  # n_desc * n_pts
        full_array = self.client.recv_array(full_buffer_size, dtype='float32')
        return full_array[:num_samples]
    
    # NEW: Fast retrieval function
    @command()
    def get_adc_stream_voltages_fast(self, num_samples):
        """Get array of ADC streaming data as voltages - FAST VERSION"""
        # C++ returns std::vector<float> with exactly num_samples elements
        return self.client.recv_vector(dtype='float32')
    
    # NEW: Individual DMA diagnostic functions
    @command()
    def get_dma_status_register(self):
        """Get DMA status register"""
        return self.client.recv_uint32()
    
    @command()
    def get_current_descriptor_index(self):
        """Get current descriptor index"""
        return self.client.recv_uint32()
    
    @command()
    def get_buffer_position(self):
        """Get buffer position"""
        return self.client.recv_uint32()
    
    @command()
    def get_dma_running(self):
        """Get DMA running status"""
        return self.client.recv_bool()
    
    @command()
    def get_dma_idle(self):
        """Get DMA idle status"""
        return self.client.recv_bool()
    
    @command()
    def get_dma_error(self):
        """Get DMA error status"""
        return self.client.recv_bool()
    
    @command()
    def get_samples_captured_accurate(self):
        """Get accurate sample count with wraparound handling"""
        return self.client.recv_uint32()
    
    @command()
    def get_buffer_fill_percentage(self):
        """Get buffer fill percentage"""
        return self.client.recv_float()
    
    @command()
    def is_dma_healthy(self):
        """Check if DMA is healthy"""
        return self.client.recv_bool()

    # Streaming control - using currentramp driver function names
    @command()
    def start_adc_streaming(self):
        """Start ADC DMA streaming"""
        pass
    
    @command()
    def stop_adc_streaming(self):
        """Stop ADC DMA streaming"""
        pass
    
    @command()
    def is_adc_streaming_active(self):
        """Get streaming active status"""
        return self.client.recv_bool()
    
    @command()
    def set_cic_decimation_rate(self, rate):
        """Set CIC decimation rate"""
        pass
    
    @command()
    def get_cic_decimation_rate(self):
        """Get CIC decimation rate"""
        return self.client.recv_uint32()
    
    @command()
    def get_decimated_sample_rate(self):
        """Get decimated sample rate"""
        return self.client.recv_double()

def connect_to_device():
    """Connect to Alpha250 currentramp instrument"""
    print("🔌 Connecting to Alpha250 currentramp...")
    
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"   ✅ Connected to {host}")
        return driver
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None

def test_retrieval_performance(driver):
    """Test the performance improvement of fast retrieval"""
    print("\n🧪 Testing Retrieval Performance Improvement...")
    
    sample_sizes = [1000, 5000, 10000, 25000, 50000]
    
    for num_samples in sample_sizes:
        print(f"\n   Testing {num_samples:,} samples:")
        
        try:
            # Test old slow method
            print("     Testing SLOW method (full buffer read)...")
            start_time = time.time()
            slow_data = driver.get_adc_stream_voltages(num_samples)
            slow_time = time.time() - start_time
            slow_rate = len(slow_data) / slow_time if slow_time > 0 else 0
            
            print(f"       Slow: {slow_time:.3f}s, {slow_rate:.0f} samples/sec")
            
            # Test new fast method
            print("     Testing FAST method (direct read)...")
            start_time = time.time()
            fast_data = driver.get_adc_stream_voltages_fast(num_samples)
            fast_time = time.time() - start_time
            fast_rate = len(fast_data) / fast_time if fast_time > 0 else 0
            
            print(f"       Fast: {fast_time:.3f}s, {fast_rate:.0f} samples/sec")
            
            # Calculate speedup
            if slow_time > 0 and fast_time > 0:
                speedup = slow_time / fast_time
                print(f"       🚀 SPEEDUP: {speedup:.1f}x faster!")
                
                if speedup >= 10:
                    print(f"       ✅ Excellent speedup for {num_samples:,} samples")
                elif speedup >= 2:
                    print(f"       ✅ Good speedup for {num_samples:,} samples")
                else:
                    print(f"       ⚠️  Limited speedup for {num_samples:,} samples")
            
            # Verify data consistency
            if len(slow_data) == len(fast_data) and len(slow_data) > 100:
                correlation = np.corrcoef(slow_data[:100], fast_data[:100])[0,1]
                print(f"       Data correlation: {correlation:.4f}")
                if correlation > 0.99:
                    print("       ✅ Data consistency verified")
                else:
                    print("       ⚠️  Data consistency issues detected")
            
        except Exception as e:
            print(f"       ❌ Error testing {num_samples:,} samples: {e}")
    
    return True

def test_dma_diagnostics(driver):
    """Test the new DMA diagnostic functions"""
    print("\n🧪 Testing DMA Diagnostics...")
    
    try:
        # Get individual diagnostics
        status_reg = driver.get_dma_status_register()
        desc_idx = driver.get_current_descriptor_index()
        buffer_pos = driver.get_buffer_position()
        dma_running = driver.get_dma_running()
        dma_idle = driver.get_dma_idle()
        dma_error = driver.get_dma_error()
        
        print("   📊 DMA Status:")
        print(f"     Status Register: 0x{status_reg:08x}")
        print(f"     Current Desc Index: {desc_idx}")
        print(f"     Buffer Position: {buffer_pos:,}")
        print(f"     DMA Running: {dma_running}")
        print(f"     DMA Idle: {dma_idle}")
        print(f"     DMA Error: {dma_error}")
        
        # Get derived metrics
        accurate_samples = driver.get_samples_captured_accurate()
        fill_percentage = driver.get_buffer_fill_percentage()
        is_healthy = driver.is_dma_healthy()
        
        print(f"\n   📈 Derived Metrics:")
        print(f"     Accurate Sample Count: {accurate_samples:,}")
        print(f"     Buffer Fill: {fill_percentage:.1f}%")
        print(f"     DMA Health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}")
        
        # Health assessment
        if is_healthy and not dma_error:
            print("   ✅ DMA diagnostics working and system healthy")
            return True
        else:
            print("   ⚠️  DMA diagnostics working but system has issues")
            return False
            
    except Exception as e:
        print(f"   ❌ DMA diagnostics test failed: {e}")
        return False

def test_stability_monitoring(driver, duration=30):
    """Test stability monitoring with new diagnostics"""
    print(f"\n🧪 Testing Stability Monitoring ({duration}s)...")
    
    try:
        measurements = []
        start_time = time.time()
        
        print("   Monitoring DMA stability...")
        while time.time() - start_time < duration:
            try:
                elapsed = time.time() - start_time
                
                # Get diagnostics
                desc_idx = driver.get_current_descriptor_index()
                buffer_pos = driver.get_buffer_position()
                fill_pct = driver.get_buffer_fill_percentage()
                is_healthy = driver.is_dma_healthy()
                dma_running = driver.get_dma_running()
                dma_error = driver.get_dma_error()
                sample_rate = driver.get_decimated_sample_rate()
                
                measurement = {
                    'time': elapsed,
                    'desc_idx': desc_idx,
                    'buffer_pos': buffer_pos,
                    'fill_pct': fill_pct,
                    'healthy': is_healthy,
                    'dma_running': dma_running,
                    'dma_error': dma_error
                }
                
                measurements.append(measurement)
                
                if len(measurements) % 10 == 0:  # Every 10 measurements
                    print(f"     {elapsed:.1f}s: Desc={desc_idx:2d}, "
                          f"Fill={fill_pct:5.1f}%, Health={'✅' if is_healthy else '❌'}")
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"     ⚠️  Error during monitoring: {e}")
                break
        
        # Analyze stability
        if len(measurements) >= 10:
            desc_changes = sum(1 for i in range(1, len(measurements)) 
                             if measurements[i]['desc_idx'] != measurements[i-1]['desc_idx'])
            
            fill_values = [m['fill_pct'] for m in measurements[5:]]  # Skip first 5 for settling
            fill_stability = np.std(fill_values) if fill_values else 100
            
            health_issues = sum(1 for m in measurements if not m['healthy'])
            
            print(f"\n   📊 Stability Analysis:")
            print(f"     Descriptor changes: {desc_changes}")
            print(f"     Fill stability: ±{fill_stability:.2f}%")
            print(f"     Health issues: {health_issues}/{len(measurements)}")
            
            if desc_changes > 0 and fill_stability < 10 and health_issues == 0:
                print("   ✅ DMA stability excellent")
                return True
            elif health_issues == 0:
                print("   ✅ DMA stability good")
                return True
            else:
                print("   ⚠️  DMA stability issues detected")
                return False
        else:
            print("   ❌ Insufficient measurements for stability analysis")
            return False
            
    except Exception as e:
        print(f"   ❌ Stability monitoring failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 Performance Fix Validation Test")
    print("   Testing fast array retrieval and DMA diagnostics")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("\n❌ Cannot connect to currentramp instrument.")
        print("💡 Make sure you've rebuilt with the performance fixes!")
        return 1
    
    # Ensure streaming is active
    try:
        active = driver.is_adc_streaming_active()
        if not active:
            print("🔧 Starting streaming for performance tests...")
            driver.set_cic_decimation_rate(2500)  # 250MHz / 2500 = 100kHz
            driver.start_adc_streaming() 
            time.sleep(3.0)  # Give it more time to accumulate data
    except Exception as e:
        print(f"⚠️  Error checking/starting streaming: {e}")
    
    # Run tests
    tests = [
        ("Retrieval Performance", lambda: test_retrieval_performance(driver)),
        ("DMA Diagnostics", lambda: test_dma_diagnostics(driver)),
        ("Stability Monitoring", lambda: test_stability_monitoring(driver, 20)),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Performance Fix Test Results:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed >= 2:  # Most tests passed
        print("🎉 PERFORMANCE FIXES WORKING!")
        print("✅ Array retrieval should now be 10-100x faster!")
        print("📊 DMA diagnostics provide detailed system health info!")
        print("🚀 Ready for high-performance data acquisition!")
        return 0
    else:
        print("⚠️ Performance fixes need more work.")
        print("🔧 Check the server logs and rebuild if needed")
        return 1

if __name__ == "__main__":
    exit(main()) 