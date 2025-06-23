#!/usr/bin/env python3
"""
Graph ADC Data Over Time
Simple script to capture and plot ADC data over a specified time period
Uses the optimized DMA streaming for fast, continuous data acquisition
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from koheron import connect, command

class CurrentRamp:
    """Python interface to CurrentRamp driver"""
    
    def __init__(self, client):
        self.client = client

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
        """Check if streaming is active"""
        return self.client.recv_bool()
    
    @command()
    def set_cic_decimation_rate(self, rate):
        """Set CIC decimation rate"""
        pass
    
    @command()
    def get_decimated_sample_rate(self):
        """Get actual sample rate"""
        return self.client.recv_double()
    
    @command()
    def get_adc_stream_voltages_fast(self, num_samples):
        """Get ADC data - OPTIMIZED VERSION (85-2,126x faster!)"""
        return self.client.recv_vector(dtype='float32')
    
    @command()
    def get_current_descriptor_index(self):
        """Get current DMA descriptor for monitoring"""
        return self.client.recv_uint32()

def connect_to_device():
    """Connect to Alpha250"""
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"✅ Connected to {host}")
        return driver
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None

def capture_adc_data(driver, duration_seconds=10, sample_rate_khz=100):
    """
    Capture ADC data over specified time period
    
    Args:
        driver: CurrentRamp driver instance
        duration_seconds: How long to capture (e.g., 10 seconds)
        sample_rate_khz: Sample rate in kHz (30.5, 50, 100, 200)
    
    Returns:
        timestamps, voltage_data
    """
    print(f"🎯 Capturing ADC data for {duration_seconds} seconds at {sample_rate_khz}kHz...")
    
    # Set up decimation rate for desired sample rate
    decimation_rates = {
        30.5: 8192,   # 250MHz / 8192 = 30.5kHz
        50: 5000,     # 250MHz / 5000 = 50kHz  
        100: 2500,    # 250MHz / 2500 = 100kHz
        200: 1250,    # 250MHz / 1250 = 200kHz
    }
    
    if sample_rate_khz not in decimation_rates:
        print(f"⚠️  Unsupported sample rate {sample_rate_khz}kHz, using 100kHz")
        sample_rate_khz = 100
    
    # Configure and start streaming
    driver.set_cic_decimation_rate(decimation_rates[sample_rate_khz])
    time.sleep(0.5)
    
    actual_fs = driver.get_decimated_sample_rate()
    print(f"   Actual sample rate: {actual_fs:.0f} Hz")
    
    # Start streaming if not already active
    if not driver.is_adc_streaming_active():
        driver.start_adc_streaming()
        time.sleep(2.0)  # Let buffer fill
    
    # Calculate how many samples we need
    total_samples = int(actual_fs * duration_seconds)
    print(f"   Target samples: {total_samples:,}")
    
    # Capture data in chunks to avoid memory issues
    chunk_size = min(50000, total_samples)  # 50k samples per chunk
    all_data = []
    samples_captured = 0
    
    start_time = time.time()
    
    while samples_captured < total_samples:
        # Calculate remaining samples
        remaining = total_samples - samples_captured
        current_chunk = min(chunk_size, remaining)
        
        # Capture chunk using FAST method (85-2,126x speedup!)
        chunk_start = time.time()
        voltage_chunk = driver.get_adc_stream_voltages_fast(current_chunk)
        chunk_time = time.time() - chunk_start
        
        all_data.extend(voltage_chunk)
        samples_captured += len(voltage_chunk)
        
        elapsed = time.time() - start_time
        progress = samples_captured / total_samples * 100
        
        print(f"   Progress: {progress:.1f}% ({samples_captured:,}/{total_samples:,}) "
              f"- chunk: {len(voltage_chunk):,} in {chunk_time:.3f}s")
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.01)
    
    total_time = time.time() - start_time
    print(f"   ✅ Captured {len(all_data):,} samples in {total_time:.1f}s")
    
    # Create timestamps
    timestamps = np.arange(len(all_data)) / actual_fs
    
    return timestamps, np.array(all_data), actual_fs

def plot_adc_data(timestamps, voltage_data, sample_rate, duration, save_file=None):
    """Create a beautiful plot of the ADC data"""
    
    print(f"📊 Creating plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'ADC Data Capture - {duration:.1f} seconds at {sample_rate:.0f} Hz', 
                 fontsize=16, fontweight='bold')
    
    # Full time series
    ax1.plot(timestamps, voltage_data, 'b-', linewidth=0.8, alpha=0.8)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('ADC Voltage (V)')
    ax1.set_title(f'Complete Time Series ({len(voltage_data):,} samples)')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box
    stats_text = f"""Statistics:
Duration: {duration:.1f}s
Samples: {len(voltage_data):,}
Rate: {sample_rate:.0f} Hz
Range: {np.min(voltage_data):+.3f}V to {np.max(voltage_data):+.3f}V
Mean: {np.mean(voltage_data):+.3f}V
Std: {np.std(voltage_data):.3f}V"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='lightblue', alpha=0.8), fontfamily='monospace')
    
    # Zoomed view of first 2 seconds (or all data if shorter)
    zoom_duration = min(2.0, duration)
    zoom_samples = int(zoom_duration * sample_rate)
    
    if len(voltage_data) > zoom_samples:
        ax2.plot(timestamps[:zoom_samples], voltage_data[:zoom_samples], 
                'r-', linewidth=1.0)
        ax2.set_title(f'Zoomed View - First {zoom_duration} seconds')
    else:
        ax2.plot(timestamps, voltage_data, 'r-', linewidth=1.0)
        ax2.set_title('Full Data (< 2 seconds)')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('ADC Voltage (V)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if save_file is None:
        save_file = f"adc_capture_{duration:.0f}s_{sample_rate:.0f}hz.png"
    
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"💾 Plot saved as: {save_file}")
    
    # Show plot
    plt.show()
    
    return save_file

def main():
    """Main function - customize this for your needs!"""
    print("=" * 60)
    print("🎯 ADC Data Graphing Tool")
    print("   Optimized for fast, high-quality data capture")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        print("❌ Cannot connect to device")
        return 1
    
    # === CUSTOMIZE THESE PARAMETERS ===
    duration_seconds = 1       # How long to capture
    sample_rate_khz = 100      # Sample rate: 30.5, 50, 100, or 200 kHz
    
    print(f"🎯 Goal: Graph ADC input for {duration_seconds} seconds")
    print(f"📊 Sample rate: {sample_rate_khz} kHz")
    
    try:
        # Capture data
        timestamps, voltage_data, actual_fs = capture_adc_data(
            driver, duration_seconds, sample_rate_khz)
        
        # Create plot
        plot_file = plot_adc_data(timestamps, voltage_data, actual_fs, duration_seconds)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Captured {len(voltage_data):,} samples over {duration_seconds} seconds")
        print(f"📊 Plot saved as: {plot_file}")
        print(f"⚡ Used optimized DMA streaming (85-2,126x faster!)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 