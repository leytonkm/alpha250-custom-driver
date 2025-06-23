#!/usr/bin/env python3
"""
Test Both ADC Channels
Quick test to see which ADC channel shows the real 10Hz sine wave
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    def __init__(self, client):
        self.client = client

    @command()
    def select_adc_channel(self, channel):
        """Select ADC channel for streaming (0 or 1)"""
        pass
    
    @command()
    def start_adc_streaming(self):
        """Start ADC streaming"""
        pass
    
    @command()
    def get_adc_stream_voltages_fast(self, num_samples):
        """Get ADC data fast"""
        return self.client.recv_vector(dtype='float32')

def test_both_channels():
    print("🔍 Testing Both ADC Channels")
    print("=" * 50)
    
    try:
        client = connect('192.168.1.20', 'currentramp', restart=False)
        driver = CurrentRamp(client)
        
        # Test both channels
        for channel in [0, 1]:
            print(f"\n📊 Testing ADC Channel {channel}:")
            
            # Select channel and start streaming
            driver.select_adc_channel(channel)
            time.sleep(0.5)
            driver.start_adc_streaming()
            time.sleep(2.0)  # Let buffer fill
            
            # Get data
            data = driver.get_adc_stream_voltages_fast(10000)  # 10k samples
            
            # Quick analysis
            mean_val = np.mean(data)
            std_val = np.std(data)
            min_val = np.min(data)
            max_val = np.max(data)
            
            print(f"   Samples: {len(data):,}")
            print(f"   Range: {min_val:+.3f}V to {max_val:+.3f}V")
            print(f"   Mean: {mean_val:+.3f}V")
            print(f"   Std: {std_val:.3f}V")
            
            # Look for 10Hz pattern (should see ~10 cycles in 1 second at 100kHz)
            # Take first 1 second of data
            fs = 100000  # 100kHz sample rate
            one_second_samples = min(fs, len(data))
            
            if one_second_samples > 1000:
                # Check for low frequency content (10Hz sine wave)
                fft_data = np.fft.fft(data[:one_second_samples])
                freqs = np.fft.fftfreq(one_second_samples, 1/fs)
                
                # Look at power around 10Hz
                freq_10hz_idx = np.argmin(np.abs(freqs - 10))
                power_10hz = np.abs(fft_data[freq_10hz_idx])
                
                # Look at total power
                total_power = np.sum(np.abs(fft_data))
                relative_10hz_power = power_10hz / total_power * 100
                
                print(f"   10Hz power: {relative_10hz_power:.2f}% of total")
                
                if relative_10hz_power > 1.0:
                    print(f"   ✅ Possible 10Hz signal detected!")
                else:
                    print(f"   ❌ No significant 10Hz signal")
            
            # Quick plot for visual inspection
            plt.figure(figsize=(12, 4))
            
            # Plot first 0.5 seconds
            plot_samples = min(50000, len(data))  # 0.5s at 100kHz
            time_axis = np.arange(plot_samples) / 100000
            
            plt.plot(time_axis, data[:plot_samples], 'b-', linewidth=0.5)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Voltage (V)')
            plt.title(f'ADC Channel {channel} - First 0.5 seconds')
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = f"adc_channel_{channel}_test.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   📊 Plot saved: {plot_file}")
        
        print(f"\n🎯 Results:")
        print("   Check the plots to see which channel shows your 10Hz sine wave")
        print("   Look for smooth sine wave vs. high-frequency noise")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_both_channels() 