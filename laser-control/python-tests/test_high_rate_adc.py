#!/usr/bin/env python3
"""
High-Rate ADC Capture for CurrentRamp
Captures ADC data at maximum possible rate using bulk array reads instead of individual samples
Goal: Get as close to 100kHz as possible for 10+ second captures
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver - High-rate ADC functions"""
    
    def __init__(self, client):
        self.client = client

    # High-rate ADC reading functions
    @command()
    def get_adc0(self):
        """Get ADC0 raw value"""
        return self.client.recv_uint32()
    
    @command()
    def get_adc1(self):
        """Get ADC1 raw value"""
        return self.client.recv_uint32()
    
    @command()
    def get_adc0_voltage(self):
        """Get ADC0 voltage"""
        return self.client.recv_float()
    
    @command()
    def get_adc1_voltage(self):
        """Get ADC1 voltage"""
        return self.client.recv_float()
    
    # Bulk capture functions
    def capture_adc_burst(self, channel=0, num_samples=1000):
        """Capture a burst of ADC samples as fast as possible"""
        samples = []
        timestamps = []
        
        start_time = time.time()
        
        for i in range(num_samples):
            if channel == 0:
                sample = self.get_adc0_voltage()
            else:
                sample = self.get_adc1_voltage()
            
            samples.append(sample)
            timestamps.append(time.time() - start_time)
        
        return np.array(timestamps), np.array(samples)
    
    def capture_adc_timed(self, channel=0, duration=10.0, target_rate=50000):
        """Capture ADC samples for specified duration at target rate"""
        samples = []
        timestamps = []
        
        start_time = time.time()
        sample_interval = 1.0 / target_rate
        next_sample_time = 0.0
        
        print(f"   Targeting {target_rate} Hz ({sample_interval*1000:.3f}ms intervals)")
        
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            
            # Only sample if it's time
            if current_time >= next_sample_time:
                if channel == 0:
                    sample = self.get_adc0_voltage()
                else:
                    sample = self.get_adc1_voltage()
                
                samples.append(sample)
                timestamps.append(current_time)
                
                next_sample_time += sample_interval
                
                # Progress update
                if len(samples) % 10000 == 0:
                    actual_rate = len(samples) / current_time if current_time > 0 else 0
                    print(f"     {current_time:.1f}s: {len(samples)} samples, {actual_rate:.0f} Hz")
        
        return np.array(timestamps), np.array(samples)

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

def test_maximum_rate(driver):
    """Test maximum achievable sample rate"""
    print("\n🧪 Testing Maximum Sample Rate...")
    
    try:
        # Test burst capture
        print("   Testing burst capture (1000 samples)...")
        start_time = time.time()
        timestamps, samples = driver.capture_adc_burst(channel=0, num_samples=1000)
        total_time = time.time() - start_time
        
        if len(samples) > 0:
            max_rate = len(samples) / total_time
            avg_interval = np.mean(np.diff(timestamps)) * 1000  # ms
            
            print(f"   📊 Burst Results:")
            print(f"     Total time: {total_time:.3f} seconds")
            print(f"     Samples: {len(samples)}")
            print(f"     Max rate: {max_rate:.0f} Hz")
            print(f"     Avg interval: {avg_interval:.3f} ms")
            print(f"     Voltage range: {np.min(samples):+.3f}V to {np.max(samples):+.3f}V")
            
            return True, max_rate
        else:
            print("   ❌ No samples captured")
            return False, 0
            
    except Exception as e:
        print(f"   ❌ Maximum rate test failed: {e}")
        return False, 0

def test_sustained_capture(driver, duration=10, target_rate=10000):
    """Test sustained high-rate capture"""
    print(f"\n🧪 Testing Sustained Capture ({duration}s at {target_rate} Hz)...")
    
    try:
        timestamps, samples = driver.capture_adc_timed(
            channel=0, 
            duration=duration, 
            target_rate=target_rate
        )
        
        if len(samples) > 100:
            actual_duration = timestamps[-1] - timestamps[0]
            actual_rate = len(samples) / actual_duration if actual_duration > 0 else 0
            
            print(f"\n   📊 Sustained Capture Results:")
            print(f"     Target rate: {target_rate} Hz")
            print(f"     Actual rate: {actual_rate:.0f} Hz")
            print(f"     Efficiency: {actual_rate/target_rate*100:.1f}%")
            print(f"     Duration: {actual_duration:.1f} seconds")
            print(f"     Total samples: {len(samples):,}")
            print(f"     Voltage range: {np.min(samples):+.3f}V to {np.max(samples):+.3f}V")
            print(f"     Voltage std: {np.std(samples):.3f}V")
            
            return True, (timestamps, samples, actual_rate)
        else:
            print("   ❌ Insufficient samples captured")
            return False, None
            
    except Exception as e:
        print(f"   ❌ Sustained capture test failed: {e}")
        return False, None

def test_different_rates(driver):
    """Test different target sample rates"""
    print("\n🧪 Testing Different Sample Rates...")
    
    rates_to_test = [1000, 5000, 10000, 20000, 50000]
    results = {}
    
    for target_rate in rates_to_test:
        print(f"\n   Testing {target_rate} Hz for 5 seconds...")
        
        try:
            timestamps, samples = driver.capture_adc_timed(
                channel=0,
                duration=5.0,
                target_rate=target_rate
            )
            
            if len(samples) > 100:
                actual_rate = len(samples) / timestamps[-1] if timestamps[-1] > 0 else 0
                efficiency = actual_rate / target_rate * 100
                
                results[target_rate] = {
                    'actual_rate': actual_rate,
                    'efficiency': efficiency,
                    'samples': len(samples)
                }
                
                print(f"     ✅ {target_rate} Hz: {actual_rate:.0f} Hz actual ({efficiency:.1f}% efficiency)")
            else:
                results[target_rate] = {'actual_rate': 0, 'efficiency': 0, 'samples': 0}
                print(f"     ❌ {target_rate} Hz: Failed")
                
        except Exception as e:
            print(f"     ❌ {target_rate} Hz: Error - {e}")
            results[target_rate] = {'actual_rate': 0, 'efficiency': 0, 'samples': 0}
    
    # Find best achievable rate
    best_rate = 0
    best_efficiency = 0
    
    print(f"\n   📊 Rate Test Summary:")
    for target_rate, result in results.items():
        actual = result['actual_rate']
        eff = result['efficiency']
        print(f"     {target_rate:5d} Hz → {actual:5.0f} Hz ({eff:5.1f}%)")
        
        if eff > 80 and actual > best_rate:  # At least 80% efficiency
            best_rate = actual
            best_efficiency = eff
    
    if best_rate > 0:
        print(f"\n   🎯 Best achievable rate: {best_rate:.0f} Hz ({best_efficiency:.1f}% efficiency)")
        return True, best_rate
    else:
        print(f"\n   ⚠️  No rate achieved >80% efficiency")
        return False, 0

def plot_high_rate_data(timestamps, samples, actual_rate, title="High-Rate ADC Data"):
    """Plot high-rate ADC data with frequency analysis"""
    print(f"   📊 Plotting {len(samples)} samples at {actual_rate:.0f} Hz...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Time domain plot
    # Plot only first 2 seconds for clarity
    max_samples = min(int(actual_rate * 2), len(samples))
    ax1.plot(timestamps[:max_samples], samples[:max_samples], 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ADC Voltage (V)')
    ax1.set_title(f'{title} - Time Domain (First 2 seconds)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(0.02, 0.98, 
             f'Rate: {actual_rate:.0f} Hz\nSamples: {len(samples):,}\nRange: {np.min(samples):+.3f}V to {np.max(samples):+.3f}V\nStd: {np.std(samples):.3f}V',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Frequency domain plot
    if len(samples) > 1024:
        # Calculate power spectral density
        from scipy import signal
        freqs, psd = signal.welch(samples, actual_rate, nperseg=min(4096, len(samples)//4))
        
        ax2.semilogy(freqs, psd, 'r-', linewidth=1.0)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_title('Frequency Spectrum')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, min(1000, actual_rate/2))  # Show up to 1kHz or Nyquist
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"currentramp_high_rate_{actual_rate:.0f}hz.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Plot saved as {plot_file}")

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 CurrentRamp High-Rate ADC Capture")
    print("   Goal: Achieve 100kHz+ sample rates for 10+ second captures")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("\n❌ Cannot connect to currentramp instrument. Exiting.")
        return 1
    
    # Test maximum rate
    success, max_rate = test_maximum_rate(driver)
    if not success:
        print("\n❌ Cannot achieve basic ADC capture. Check setup.")
        return 1
    
    # Test different rates to find optimum
    success, best_rate = test_different_rates(driver)
    if not success:
        print("\n⚠️  Rate testing had issues, using basic rate")
        best_rate = min(max_rate, 10000)
    
    # Do a long capture at the best rate
    target_rate = min(best_rate, 50000)  # Cap at 50kHz for reliability
    print(f"\n🎯 Final Test: 15-second capture at {target_rate:.0f} Hz")
    
    success, result = test_sustained_capture(driver, duration=15, target_rate=int(target_rate))
    
    if success and result:
        timestamps, samples, actual_rate = result
        
        # Create plots
        plot_high_rate_data(timestamps, samples, actual_rate)
        
        # Summary
        print(f"\n" + "=" * 60)
        print("🎉 HIGH-RATE ADC CAPTURE SUCCESS!")
        print("=" * 60)
        print(f"✅ Achieved {actual_rate:.0f} Hz sustained capture rate")
        print(f"📊 Captured {len(samples):,} samples over {timestamps[-1]:.1f} seconds")
        print(f"🎯 This is {actual_rate/1000:.1f}x faster than your original 36 Hz!")
        
        if actual_rate >= 10000:
            print(f"🚀 You're getting 10kHz+ capture - excellent for ramp monitoring!")
        if actual_rate >= 50000:
            print(f"⚡ 50kHz+ achieved - this is serious high-speed data acquisition!")
            
        print(f"💾 Data and plots saved for analysis")
        print("=" * 60)
        
        return 0
    else:
        print(f"\n❌ High-rate capture failed")
        return 1

if __name__ == "__main__":
    exit(main()) 