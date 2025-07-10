#!/usr/bin/env python3
"""
FPGA Timing Debug Script
Investigates the rate instability issues in DMA streaming
Analyzes clock domains, CIC decimator, and DMA timing
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """Python interface for FPGA timing debugging"""
    
    def __init__(self, client):
        self.client = client

    # DMA and streaming functions
    @command()
    def get_current_descriptor_index(self):
        """Get current descriptor index"""
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
    def get_streaming_active(self):
        """Get streaming active status"""
        return self.client.recv_bool()
    
    @command()
    def start_streaming(self):
        """Start DMA streaming"""
        pass
    
    @command()
    def stop_streaming(self):
        """Stop DMA streaming"""
        pass
    
    @command()
    def is_adc_streaming_active(self):
        """Get streaming active status (alternative name)"""
        return self.client.recv_bool()
    
    @command()
    def start_adc_streaming(self):
        """Start ADC DMA streaming"""
        pass
    
    @command()
    def get_decimation_rate(self):
        """Get CIC decimation rate"""
        return self.client.recv_uint32()
    
    @command()
    def set_decimation_rate(self, rate):
        """Set CIC decimation rate"""
        pass
    
    @command()
    def get_decimated_sample_rate(self):
        """Get decimated sample rate"""
        return self.client.recv_double()
    
    @command()
    def get_samples_captured_accurate(self):
        """Get accurate sample count"""
        return self.client.recv_uint32()

def connect_to_device():
    """Connect to Alpha15 currentramp instrument"""
    print("🔌 Connecting to Alpha15 currentramp...")
    
    try:
        host = os.environ.get('HOST', '192.168.1.115')
        client = connect(host, 'alpha15-laser-control', restart=False)
        driver = CurrentRamp(client)
        print(f"   ✅ Connected to {host}")
        return driver
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return None

def analyze_descriptor_timing(driver, duration=60):
    """Analyze DMA descriptor advancement timing"""
    print(f"\n🔍 Analyzing DMA Descriptor Timing ({duration}s)...")
    
    measurements = []
    start_time = time.time()
    last_desc = -1
    desc_changes = []
    
    print("   Monitoring descriptor changes...")
    
    while time.time() - start_time < duration:
        try:
            elapsed = time.time() - start_time
            current_desc = driver.get_current_descriptor_index()
            dma_running = driver.get_dma_running()
            dma_idle = driver.get_dma_idle()
            dma_error = driver.get_dma_error()
            
            # Record descriptor changes
            if current_desc != last_desc and last_desc >= 0:
                change_time = elapsed
                if desc_changes:
                    interval = change_time - desc_changes[-1]['time']
                    desc_changes.append({
                        'time': change_time,
                        'from_desc': last_desc,
                        'to_desc': current_desc,
                        'interval': interval
                    })
                    print(f"     {change_time:.1f}s: Desc {last_desc} → {current_desc} ({interval:.2f}s interval)")
                else:
                    desc_changes.append({
                        'time': change_time,
                        'from_desc': last_desc,
                        'to_desc': current_desc,
                        'interval': 0
                    })
            
            last_desc = current_desc
            
            measurement = {
                'time': elapsed,
                'desc_idx': current_desc,
                'dma_running': dma_running,
                'dma_idle': dma_idle,
                'dma_error': dma_error
            }
            measurements.append(measurement)
            
            time.sleep(0.1)  # High resolution monitoring
            
        except Exception as e:
            print(f"     ⚠️  Error: {e}")
            break
    
    # Analysis
    if len(desc_changes) >= 3:
        intervals = [dc['interval'] for dc in desc_changes[1:]]  # Skip first
        
        print(f"\n   📊 Descriptor Timing Analysis:")
        print(f"     Total descriptor changes: {len(desc_changes)}")
        print(f"     Average interval: {np.mean(intervals):.2f}s")
        print(f"     Interval std dev: {np.std(intervals):.2f}s")
        print(f"     Min interval: {np.min(intervals):.2f}s")
        print(f"     Max interval: {np.max(intervals):.2f}s")
        
        # Expected interval calculation
        # Each descriptor holds 64K samples
        # At 100kHz decimated rate: 64K / 100kHz = 0.64s
        expected_interval = 65536 / 100000  # 0.65536s
        print(f"     Expected interval: {expected_interval:.2f}s")
        
        # Check for timing regularity
        interval_variation = np.std(intervals) / np.mean(intervals) * 100
        print(f"     Timing variation: ±{interval_variation:.1f}%")
        
        if interval_variation < 5:
            print("   ✅ Descriptor timing is regular")
        elif interval_variation < 15:
            print("   ⚠️  Descriptor timing has some variation")
        else:
            print("   ❌ Descriptor timing is highly irregular")
        
        return True, desc_changes
    else:
        print("   ❌ Insufficient descriptor changes to analyze")
        return False, []

def test_different_decimation_rates(driver):
    """Test stability at different CIC decimation rates"""
    print("\n🔍 Testing Different CIC Decimation Rates...")
    
    # Test rates: higher decimation = lower output rate = longer descriptor fill time
    test_rates = [
        (8192, "30.5 kHz"),   # Longest descriptor time: ~2.1s
        (5000, "50 kHz"),     # Medium: ~1.3s  
        (2500, "100 kHz"),    # Current default: ~0.65s
        (1250, "200 kHz"),    # Fast: ~0.33s
    ]
    
    rate_results = {}
    
    for rate, description in test_rates:
        print(f"\n   Testing rate {rate} ({description})...")
        
        try:
            # Set new rate
            driver.set_decimation_rate(rate)
            time.sleep(1.0)  # Let it settle
            
            # Verify rate
            actual_rate = driver.get_decimation_rate()
            actual_fs = driver.get_decimated_sample_rate()
            
            print(f"     Set: {rate}, Actual: {actual_rate}, Sample rate: {actual_fs:.0f} Hz")
            
            if actual_rate == rate:
                # Monitor for 30 seconds
                start_time = time.time()
                measurements = []
                last_desc = -1
                desc_changes = 0
                
                while time.time() - start_time < 30:
                    try:
                        current_desc = driver.get_current_descriptor_index()
                        
                        if current_desc != last_desc and last_desc >= 0:
                            desc_changes += 1
                        last_desc = current_desc
                        
                        measurements.append({
                            'time': time.time() - start_time,
                            'desc_idx': current_desc
                        })
                        
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"       Error: {e}")
                        break
                
                # Calculate descriptor change rate
                if len(measurements) > 0:
                    total_time = measurements[-1]['time']
                    desc_change_rate = desc_changes / total_time if total_time > 0 else 0
                    
                    rate_results[rate] = {
                        'sample_rate': actual_fs,
                        'desc_changes': desc_changes,
                        'desc_change_rate': desc_change_rate,
                        'expected_change_rate': actual_fs / 65536,  # Expected: fs / samples_per_desc
                        'measurements': len(measurements)
                    }
                    
                    print(f"     Descriptor changes: {desc_changes} in {total_time:.1f}s")
                    print(f"     Change rate: {desc_change_rate:.3f} Hz")
                    print(f"     Expected: {actual_fs / 65536:.3f} Hz")
                
            else:
                print(f"     ⚠️  Rate setting failed")
                
        except Exception as e:
            print(f"     ❌ Error testing rate {rate}: {e}")
    
    # Set back to default
    driver.set_decimation_rate(2500)
    
    # Analysis
    print(f"\n   📊 Decimation Rate Analysis:")
    for rate, result in rate_results.items():
        expected = result['expected_change_rate']
        actual = result['desc_change_rate']
        error = abs(actual - expected) / expected * 100 if expected > 0 else 100
        
        print(f"     Rate {rate:4d}: {actual:.3f} Hz (expected {expected:.3f} Hz, error {error:.1f}%)")
    
    return rate_results

def investigate_rate_instability(driver, duration=120):
    """Deep investigation of rate instability patterns"""
    print(f"\n🔍 Investigating Rate Instability ({duration}s)...")
    
    measurements = []
    start_time = time.time()
    
    print("   High-resolution monitoring...")
    
    while time.time() - start_time < duration:
        try:
            elapsed = time.time() - start_time
            
            # Get multiple measurements rapidly
            desc_idx = driver.get_current_descriptor_index()
            buffer_pos = desc_idx * 65536  # desc_idx * n_pts
            samples = driver.get_samples_captured_accurate()
            dma_running = driver.get_dma_running()
            dma_idle = driver.get_dma_idle()
            dma_error = driver.get_dma_error()
            
            measurement = {
                'time': elapsed,
                'desc_idx': desc_idx,
                'buffer_pos': buffer_pos,
                'samples': samples,
                'dma_running': dma_running,
                'dma_idle': dma_idle,
                'dma_error': dma_error
            }
            measurements.append(measurement)
            
            if len(measurements) % 100 == 0:
                print(f"     {elapsed:.1f}s: {len(measurements)} measurements")
            
            time.sleep(0.05)  # 20 Hz monitoring
            
        except Exception as e:
            print(f"     Error: {e}")
            break
    
    # Analysis of patterns
    if len(measurements) > 50:
        times = [m['time'] for m in measurements]
        desc_indices = [m['desc_idx'] for m in measurements]
        buffer_positions = [m['buffer_pos'] for m in measurements]
        
        # Look for patterns in descriptor advancement
        desc_changes = []
        for i in range(1, len(measurements)):
            if desc_indices[i] != desc_indices[i-1]:
                desc_changes.append({
                    'time': times[i],
                    'from_desc': desc_indices[i-1],
                    'to_desc': desc_indices[i]
                })
        
        print(f"\n   📊 Rate Instability Analysis:")
        print(f"     Total measurements: {len(measurements)}")
        print(f"     Descriptor changes: {len(desc_changes)}")
        
        if len(desc_changes) >= 3:
            intervals = []
            for i in range(1, len(desc_changes)):
                interval = desc_changes[i]['time'] - desc_changes[i-1]['time']
                intervals.append(interval)
            
            print(f"     Descriptor intervals:")
            print(f"       Mean: {np.mean(intervals):.2f}s")
            print(f"       Std:  {np.std(intervals):.2f}s")
            print(f"       Min:  {np.min(intervals):.2f}s")
            print(f"       Max:  {np.max(intervals):.2f}s")
            
            # Look for patterns
            if len(intervals) >= 5:
                # Check for alternating pattern
                odd_intervals = intervals[::2]
                even_intervals = intervals[1::2]
                
                if len(odd_intervals) >= 2 and len(even_intervals) >= 2:
                    odd_mean = np.mean(odd_intervals)
                    even_mean = np.mean(even_intervals)
                    
                    print(f"     Pattern Analysis:")
                    print(f"       Odd intervals:  {odd_mean:.2f}s")
                    print(f"       Even intervals: {even_mean:.2f}s")
                    
                    if abs(odd_mean - even_mean) > 0.1:
                        print("       ⚠️  ALTERNATING PATTERN DETECTED!")
                        print("       This suggests burst vs. slow filling")
                    else:
                        print("       ✅ No significant alternating pattern")
        
        return True, measurements
    else:
        print("   ❌ Insufficient data for analysis")
        return False, []

def main():
    """Main debugging function"""
    print("=" * 60)
    print("🔍 FPGA Timing Debug Script")
    print("   Investigating DMA rate instability")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("\n❌ Cannot connect to device")
        return 1
    
    # Ensure streaming is active
    try:
        active = driver.is_adc_streaming_active()
        if not active:
            print("🔧 Starting streaming for debug...")
            driver.set_decimation_rate(2500)  # 100kHz
            driver.start_adc_streaming()
            time.sleep(3.0)  # Give it more time to start properly
            
            # Verify it started
            active = driver.is_adc_streaming_active()
            if active:
                print("   ✅ Streaming started successfully")
            else:
                print("   ❌ Failed to start streaming")
                return 1
        else:
            print("   ✅ Streaming already active")
    except Exception as e:
        print(f"⚠️  Error starting streaming: {e}")
        return 1
    
    # Run debug tests
    tests = [
        ("Descriptor Timing", lambda: analyze_descriptor_timing(driver, 60)),
        ("Decimation Rates", lambda: test_different_decimation_rates(driver)),
        ("Rate Instability", lambda: investigate_rate_instability(driver, 90)),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        try:
            result = test_func()
            if isinstance(result, tuple):
                results[test_name] = result[0]
            else:
                results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FPGA Timing Debug Results:")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL" 
        print(f"   {test_name:20} {status}")
    
    print(f"\n🎯 Debug Complete")
    print("💡 Check the analysis above for timing patterns and irregularities")
    print("🔧 Look for alternating patterns that suggest burst/slow cycles")
    
    return 0

if __name__ == "__main__":
    exit(main()) 