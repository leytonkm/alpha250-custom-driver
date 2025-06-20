#!/usr/bin/env python3
"""
CurrentRamp Data Analysis
Simple analysis tools for voltage ramp experiments
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import argparse

def analyze_ramp_response(data_file):
    """Analyze ADC response to voltage ramps"""
    print(f"📊 Analyzing ramp data from {data_file}")
    
    # This is a placeholder - in practice, you'd save data from the stream
    # For now, let's create synthetic data to show the concept
    
    # Generate synthetic ramp data
    fs = 10000  # 10 kHz sample rate
    duration = 30  # 30 seconds
    ramp_freq = 1.0  # 1 Hz ramp
    
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simulate voltage ramp (sawtooth)
    ramp_voltage = 1.25 + 1.0 * signal.sawtooth(2 * np.pi * ramp_freq * t, width=1.0)
    
    # Simulate ADC response with some noise and nonlinearity
    adc_response = ramp_voltage + 0.02 * np.random.randn(len(t))
    adc_response += 0.1 * np.sin(2 * np.pi * 60 * t)  # 60 Hz interference
    
    # Analysis
    print(f"   Data duration: {duration} seconds")
    print(f"   Sample rate: {fs} Hz")
    print(f"   Total samples: {len(t):,}")
    print(f"   Ramp frequency: {ramp_freq} Hz")
    
    # Calculate ramp statistics
    ramp_cycles = int(duration * ramp_freq)
    samples_per_cycle = len(t) // ramp_cycles
    
    print(f"   Ramp cycles: {ramp_cycles}")
    print(f"   Samples per cycle: {samples_per_cycle}")
    
    # Voltage statistics
    print(f"   Voltage range: {np.min(adc_response):.3f} to {np.max(adc_response):.3f} V")
    print(f"   Voltage mean: {np.mean(adc_response):.3f} V")
    print(f"   Voltage std: {np.std(adc_response):.3f} V")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('CurrentRamp Data Analysis', fontsize=14, fontweight='bold')
    
    # Full time series
    axes[0].plot(t, adc_response, 'b-', linewidth=0.5, alpha=0.7)
    axes[0].set_ylabel('ADC Voltage (V)')
    axes[0].set_title('Full Time Series')
    axes[0].grid(True, alpha=0.3)
    
    # Zoom into first few cycles
    zoom_samples = min(5 * samples_per_cycle, len(t))
    axes[1].plot(t[:zoom_samples], adc_response[:zoom_samples], 'b-', linewidth=1.0)
    axes[1].set_ylabel('ADC Voltage (V)')
    axes[1].set_title('First Few Ramp Cycles (Zoomed)')
    axes[1].grid(True, alpha=0.3)
    
    # Frequency spectrum
    freqs, psd = signal.welch(adc_response, fs, nperseg=4096)
    axes[2].semilogy(freqs, psd, 'r-', linewidth=1.0)
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Power Spectral Density')
    axes[2].set_title('Frequency Spectrum')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 100)  # Focus on low frequencies
    
    plt.tight_layout()
    plt.show()
    
    return t, adc_response

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(
        description='CurrentRamp Data Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data-file', default='ramp_data.npy',
                       help='Data file to analyze')
    
    args = parser.parse_args()
    
    print("🔬 CurrentRamp Data Analysis Tool")
    print("=" * 50)
    
    try:
        t, data = analyze_ramp_response(args.data_file)
        print("\n✅ Analysis complete!")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
