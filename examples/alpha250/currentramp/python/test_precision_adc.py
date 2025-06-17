#!/usr/bin/env python3
"""
Precision ADC Test for CurrentRamp
Test the 24-bit precision ADC (AD7124-8) with sine wave input
"""

import os
import time
import numpy as np
from koheron import connect, command
import matplotlib.pyplot as plt

class CurrentRamp:
    """CurrentRamp Python interface for precision ADC testing"""
    
    def __init__(self, client):
        self.client = client

    @command()
    def get_photodiode_precision(self, channel):
        """Get precision ADC reading from specified channel (0-7)"""
        return self.client.recv_float()

def test_precision_adc_sine_wave():
    """Test precision ADC with sine wave input"""
    print("🔬 Precision ADC Sine Wave Test")
    print("=" * 40)
    print("Expected: -0.5V to +0.5V sine wave at 1Hz on precision ADC channel 0")
    print("ADC specs: 24-bit, ±1.25V differential range")
    print()
    
    # Connect to device
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        print(f"Connecting to {host}...")
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print("✅ Connected successfully")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # Test precision ADC channel 0 (where sine wave is connected)
    print(f"\n📊 Reading Precision ADC Channel 0 (sine wave input):")
    
    # High-resolution sampling: 1000 samples over 1 second for 1Hz sine wave
    # This gives us 1000 points for one complete sine wave cycle
    duration = 1.0  # seconds (1 full cycle)
    sample_rate = 1000  # Hz (1000 samples per second = 1ms intervals)
    num_samples = int(duration * sample_rate)
    
    print(f"  Collecting {num_samples} samples over {duration}s at {sample_rate}Hz")
    print(f"  This gives {sample_rate} points for one complete 1Hz sine wave cycle")
    print("  Sampling every 1ms for smooth waveform capture")
    print()
    
    timestamps = []
    voltages = []
    
    start_time = time.time()
    
    for i in range(num_samples):
        try:
            voltage = driver.get_photodiode_precision(0)  # Channel 0
            current_time = time.time() - start_time
            
            timestamps.append(current_time)
            voltages.append(voltage)
            
            # Print every 100th sample to show progress without spam
            if i % 100 == 0:
                print(f"  Sample {i+1:4d}/{num_samples}: t={current_time:.3f}s, V={voltage:.4f}V")
            
            # Wait for next sample (1ms intervals)
            time.sleep(1.0 / sample_rate)
            
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"❌ Error reading sample {i+1}: {e}")
            break
    
    # Convert to numpy arrays for analysis
    timestamps = np.array(timestamps)
    voltages = np.array(voltages)
    
    if len(voltages) == 0:
        print("❌ No data collected")
        return
    
    print(f"\n📈 Analysis Results:")
    print(f"  Samples collected: {len(voltages)}")
    print(f"  Voltage range: {np.min(voltages):.4f}V to {np.max(voltages):.4f}V")
    print(f"  Peak-to-peak: {np.max(voltages) - np.min(voltages):.4f}V")
    print(f"  Mean voltage: {np.mean(voltages):.4f}V")
    print(f"  RMS voltage: {np.sqrt(np.mean(voltages**2)):.4f}V")
    print(f"  Standard deviation: {np.std(voltages):.4f}V")
    
    # Check if it looks like a sine wave (updated for 0.5V amplitude)
    expected_amplitude = 0.5  # ±0.5V sine wave
    expected_rms = expected_amplitude / np.sqrt(2)  # ~0.354V for sine wave
    
    measured_amplitude = (np.max(voltages) - np.min(voltages)) / 2
    measured_rms = np.sqrt(np.mean(voltages**2))
    
    print(f"\n🎯 Sine Wave Analysis:")
    print(f"  Expected amplitude: ±{expected_amplitude:.1f}V")
    print(f"  Measured amplitude: ±{measured_amplitude:.3f}V")
    print(f"  Expected RMS: {expected_rms:.3f}V")
    print(f"  Measured RMS: {measured_rms:.3f}V")
    
    # Amplitude check (updated tolerance for 0.5V)
    if abs(measured_amplitude - expected_amplitude) < 0.05:
        print("  ✅ Amplitude matches expected sine wave")
    else:
        print("  ⚠️  Amplitude differs from expected")
    
    # RMS check  
    if abs(measured_rms - expected_rms) < 0.05:
        print("  ✅ RMS matches expected sine wave")
    else:
        print("  ⚠️  RMS differs from expected")
    
    # Check for variation (should not be constant)
    if np.std(voltages) > 0.05:
        print("  ✅ Signal shows good variation (not stuck)")
    else:
        print("  ❌ Signal appears constant or stuck")
    
    # Frequency analysis (improved with high-resolution data)
    if len(timestamps) > 100:
        # Find zero crossings to estimate frequency
        zero_crossings = []
        for i in range(1, len(voltages)):
            if (voltages[i-1] < 0 and voltages[i] >= 0) or (voltages[i-1] >= 0 and voltages[i] < 0):
                # Linear interpolation to find exact crossing time
                if abs(voltages[i] - voltages[i-1]) > 1e-6:  # Avoid division by zero
                    t_cross = timestamps[i-1] + (timestamps[i] - timestamps[i-1]) * (-voltages[i-1]) / (voltages[i] - voltages[i-1])
                    zero_crossings.append(t_cross)
        
        if len(zero_crossings) >= 4:  # Need at least 2 full cycles
            # Calculate period from zero crossings (2 crossings per period)
            periods = []
            for i in range(2, len(zero_crossings), 2):
                period = zero_crossings[i] - zero_crossings[i-2]
                periods.append(period)
            
            if periods:
                avg_period = np.mean(periods)
                measured_freq = 1.0 / avg_period
                print(f"  Measured frequency: {measured_freq:.3f}Hz (expected: 1.000Hz)")
                print(f"  Zero crossings found: {len(zero_crossings)}")
                
                if abs(measured_freq - 1.0) < 0.05:
                    print("  ✅ Frequency matches expected 1Hz")
                else:
                    print("  ⚠️  Frequency differs from expected")
    
    # FFT analysis for frequency content
    if len(voltages) > 256:
        print(f"\n🔍 FFT Analysis:")
        # Remove DC component for cleaner FFT
        voltages_ac = voltages - np.mean(voltages)
        
        # Compute FFT
        fft = np.fft.fft(voltages_ac)
        freqs = np.fft.fftfreq(len(voltages_ac), 1.0/sample_rate)
        
        # Find peak frequency (positive frequencies only)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        peak_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC component
        peak_freq = positive_freqs[peak_idx]
        
        print(f"  Peak frequency from FFT: {peak_freq:.3f}Hz")
        if abs(peak_freq - 1.0) < 0.05:
            print("  ✅ FFT confirms 1Hz sine wave")
        else:
            print("  ⚠️  FFT shows different frequency")
    
    # Plot results
    print(f"\n📊 Plotting results...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time domain plot - full data (1 complete cycle)
    ax1.plot(timestamps, voltages, 'b-', linewidth=1.5, label='Measured', alpha=0.9)
    ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Expected ±0.5V')
    ax1.axhline(y=-0.5, color='r', linestyle='--', alpha=0.7)
    ax1.axhline(y=0.0, color='k', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Precision ADC Channel 0 - Complete 1Hz Cycle (1000 points)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add expected sine wave for comparison
    if len(timestamps) > 10:
        expected_sine = 0.5 * np.sin(2 * np.pi * 1.0 * timestamps)  # 0.5V amplitude, 1Hz
        ax1.plot(timestamps, expected_sine, 'r--', alpha=0.6, linewidth=2, label='Expected 1Hz sine')
        ax1.legend()
    
    # Time domain plot - first quarter cycle (detailed view)
    if len(timestamps) > 100:
        # Show first 0.25 seconds for very detailed view
        zoom_mask = timestamps <= 0.25
        ax2.plot(timestamps[zoom_mask], voltages[zoom_mask], 'b-', linewidth=2, label='Measured')
        if len(timestamps) > 10:
            ax2.plot(timestamps[zoom_mask], expected_sine[zoom_mask], 'r--', alpha=0.7, linewidth=2, label='Expected')
        ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
        ax2.axhline(y=-0.5, color='r', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.0, color='k', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_title('First Quarter Cycle (0-0.25s) - Ultra Detailed')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    # Histogram
    ax3.hist(voltages, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Count')
    ax3.set_title('Voltage Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Add expected distribution markers
    ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Expected peaks')
    ax3.axvline(x=-0.5, color='r', linestyle='--', alpha=0.7)
    ax3.axvline(x=0.0, color='k', linestyle='-', alpha=0.3, label='Zero')
    ax3.legend()
    
    # FFT plot
    if len(voltages) > 256:
        voltages_ac = voltages - np.mean(voltages)
        fft = np.fft.fft(voltages_ac)
        freqs = np.fft.fftfreq(len(voltages_ac), 1.0/sample_rate)
        
        # Plot positive frequencies up to 10Hz
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft[:len(fft)//2])
        
        freq_mask = positive_freqs <= 10
        ax4.plot(positive_freqs[freq_mask], positive_fft[freq_mask], 'g-', linewidth=2)
        ax4.axvline(x=1.0, color='r', linestyle='--', alpha=0.7, label='Expected 1Hz')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Magnitude')
        ax4.set_title('FFT - Frequency Content')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print(f"\n📋 Test Summary:")
    if abs(measured_amplitude - expected_amplitude) < 0.05 and np.std(voltages) > 0.05:
        print("🎉 Precision ADC appears to be working correctly!")
        print("   Your 0.5V sine wave input is being read properly.")
        print(f"   High-resolution capture: {len(voltages)} samples over {duration}s")
        print("   Perfect for smooth waveform analysis - 1 sample per millisecond!")
        print("   You can now use this for your photodiode measurements.")
    else:
        print("⚠️  Check your sine wave generator and connections:")
        print("   - Ensure sine wave is -0.5V to +0.5V amplitude")
        print("   - Verify connection to precision ADC channel 0")
        print("   - Check that sine wave frequency is 1Hz")

def test_all_precision_channels():
    """Quick test of all precision ADC channels"""
    print("\n🔍 Testing All Precision ADC Channels:")
    print("=" * 40)
    
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        
        for channel in range(8):
            voltage = driver.get_photodiode_precision(channel)
            print(f"  Channel {channel}: {voltage:.4f}V")
            time.sleep(0.1)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Run main sine wave test
    test_precision_adc_sine_wave()
    
    # Optionally test all channels
    print("\n" + "="*50)
    test_all_precision_channels() 