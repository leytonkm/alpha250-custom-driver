#!/usr/bin/env python3
"""
ADC Input Test for CurrentRamp
Tests ADC functionality through loopback connection and BRAM data capture
Provides live graphing of paired ADC/DAC data
"""

import os
import time
import numpy as np
from koheron import connect, command
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

class CurrentRamp:
    """CurrentRamp Python interface focused on ADC/BRAM functionality"""
    
    def __init__(self, client):
        self.client = client
        
        # Get BRAM sizes
        self.adc_samples_size = self.get_adc_samples_size()
        self.dac_samples_size = self.get_dac_samples_size()
        
        print(f"ADC BRAM size: {self.adc_samples_size} samples")
        print(f"DAC BRAM size: {self.dac_samples_size} samples")

    # === BRAM Data Acquisition Functions ===
    
    @command()
    def trigger_acquisition(self):
        """Trigger BRAM data acquisition"""
        pass
    
    @command()
    def get_adc_samples_size(self):
        """Get ADC BRAM size"""
        return self.client.recv_uint32()
    
    @command()
    def get_dac_samples_size(self):
        """Get DAC BRAM size"""
        return self.client.recv_uint32()
    
    @command()
    def get_adc_samples_1000(self):
        """Get 1000 ADC samples from BRAM"""
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def get_dac_samples_1000(self):
        """Get 1000 DAC samples from BRAM"""
        return self.client.recv_array(1000, dtype='uint32')
    
    @command()
    def is_acquisition_complete(self):
        """Check if acquisition is complete"""
        return self.client.recv_bool()
    
    @command()
    def get_acquisition_count(self):
        """Get acquisition count"""
        return self.client.recv_uint32()

    # === Ramp Control Functions ===
    
    @command()
    def set_ramp_frequency(self, frequency):
        """Set ramp frequency in Hz (0.001-1000)"""
        pass
    
    @command()
    def set_ramp_amplitude(self, amplitude):
        """Set ramp amplitude in V (0-2.5)"""
        pass
    
    @command()
    def set_ramp_offset(self, offset):
        """Set ramp DC offset in V (0-2.5)"""
        pass
    
    @command()
    def start_ramp(self):
        """Start hardware ramp output"""
        pass
    
    @command()
    def stop_ramp(self):
        """Stop hardware ramp output"""
        pass
    
    @command()
    def get_ramp_enabled(self):
        """Get ramp enable status"""
        return self.client.recv_bool()

    # === Photodiode/ADC Reading Functions ===
    
    @command()
    def get_photodiode_voltage(self):
        """Get photodiode voltage reading from fast ADC"""
        return self.client.recv_float()
    
    @command()
    def get_photodiode_raw(self):
        """Get raw photodiode ADC reading"""
        return self.client.recv_int16()
    
    @command()
    def set_photodiode_channel(self, channel):
        """Set photodiode ADC channel"""
        pass
    
    @command()
    def get_photodiode_channel(self):
        """Get current photodiode ADC channel"""
        return self.client.recv_uint32()

    # === Data Processing Functions ===
    
    def get_paired_data(self, n_samples=1000):
        """Get paired ADC/DAC data from BRAM"""
        self.trigger_acquisition()
        
        # Wait for acquisition to complete
        timeout = 100  # 100ms timeout
        start_time = time.time()
        while not self.is_acquisition_complete():
            time.sleep(0.001)
            if (time.time() - start_time) > timeout/1000:
                print("Warning: Acquisition timeout")
                break
        
        # Get the data
        if n_samples <= 1000:
            adc_raw = self.get_adc_samples_1000()[:n_samples]
            dac_raw = self.get_dac_samples_1000()[:n_samples]
        else:
            print(f"Warning: Requested {n_samples} samples, but only 1000 available")
            adc_raw = self.get_adc_samples_1000()
            dac_raw = self.get_dac_samples_1000()
        
        return self.convert_raw_data(adc_raw, dac_raw)
    
    def convert_raw_data(self, adc_raw, dac_raw):
        """Convert raw BRAM data to voltage values"""
        # Convert ADC data (assuming 16-bit signed, ±1.8V range)
        adc_ch0 = ((np.array(adc_raw, dtype=np.int32) & 0xFFFF) - 32768) / 32768.0 * 1.8
        adc_ch1 = (((np.array(adc_raw, dtype=np.int32) >> 16) & 0xFFFF) - 32768) / 32768.0 * 1.8
        
        # Convert DAC data (assuming 16-bit unsigned, 0-2.5V range)
        dac_ch0 = (np.array(dac_raw, dtype=np.uint32) & 0xFFFF) / 65535.0 * 2.5
        dac_ch1 = ((np.array(dac_raw, dtype=np.uint32) >> 16) & 0xFFFF) / 65535.0 * 2.5
        
        return {
            'adc': [adc_ch0, adc_ch1],
            'dac': [dac_ch0, dac_ch1],
            'time': np.arange(len(adc_ch0)) / 250e6  # Assuming 250 MHz sampling
        }

def test_adc_basic_functionality(driver):
    """Test basic ADC functionality"""
    print("\n=== Testing Basic ADC Functionality ===")
    
    try:
        # Test single ADC readings
        print("Testing single ADC readings...")
        for i in range(5):
            voltage = driver.get_photodiode_voltage()
            raw = driver.get_photodiode_raw()
            print(f"  Reading {i+1}: {voltage:.4f}V (raw: {raw})")
            time.sleep(0.1)
        
        print("✅ Basic ADC readings working")
        return True
        
    except Exception as e:
        print(f"❌ Basic ADC test failed: {e}")
        return False

def test_loopback_connection(driver):
    """Test loopback connection with known ramp output"""
    print("\n=== Testing Loopback Connection ===")
    
    try:
        # Configure a slow ramp for easy verification
        driver.set_ramp_frequency(1.0)  # 1 Hz
        driver.set_ramp_amplitude(1.0)  # 1V amplitude
        driver.set_ramp_offset(1.5)     # 1.5V offset
        
        print("Starting ramp output...")
        driver.start_ramp()
        time.sleep(0.5)  # Let ramp stabilize
        
        # Test with BRAM capture
        print("Capturing BRAM data...")
        data = driver.get_paired_data(1000)
        
        # Stop ramp
        driver.stop_ramp()
        
        # Analyze data
        dac_output = data['dac'][0]  # DAC channel 0 (ramp output)
        adc_input = data['adc'][0]   # ADC channel 0 (should match DAC)
        
        # Calculate correlation
        correlation = np.corrcoef(dac_output, adc_input)[0, 1]
        
        print(f"DAC output range: {np.min(dac_output):.3f}V to {np.max(dac_output):.3f}V")
        print(f"ADC input range: {np.min(adc_input):.3f}V to {np.max(adc_input):.3f}V")
        print(f"Correlation coefficient: {correlation:.4f}")
        
        if correlation > 0.8:
            print("✅ Loopback connection working well")
            return True, data
        elif correlation > 0.5:
            print("⚠️  Loopback connection partially working")
            return True, data
        else:
            print("❌ Loopback connection not working properly")
            return False, data
            
    except Exception as e:
        print(f"❌ Loopback test failed: {e}")
        driver.stop_ramp()  # Ensure ramp is stopped
        return False, None

def plot_paired_data(data, title="ADC/DAC Paired Data"):
    """Plot paired ADC/DAC data"""
    if data is None:
        print("No data to plot")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    time_us = data['time'] * 1e6  # Convert to microseconds
    
    # Plot DAC outputs
    ax1.plot(time_us, data['dac'][0], 'b-', label='DAC Ch0 (Ramp Output)', linewidth=2)
    ax1.plot(time_us, data['dac'][1], 'b--', label='DAC Ch1', alpha=0.7)
    ax1.set_ylabel('DAC Voltage (V)')
    ax1.set_title(f'{title} - DAC Outputs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ADC inputs
    ax2.plot(time_us, data['adc'][0], 'r-', label='ADC Ch0 (Photodiode)', linewidth=2)
    ax2.plot(time_us, data['adc'][1], 'r--', label='ADC Ch1', alpha=0.7)
    ax2.set_ylabel('ADC Voltage (V)')
    ax2.set_title(f'{title} - ADC Inputs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot overlay for loopback comparison
    ax3.plot(time_us, data['dac'][0], 'b-', label='DAC Ch0 Output', linewidth=2)
    ax3.plot(time_us, data['adc'][0], 'r-', label='ADC Ch0 Input', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Voltage (V)')
    ax3.set_title(f'{title} - Loopback Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def live_plot_demo(driver, duration=10):
    """Demonstrate live plotting capability"""
    print(f"\n=== Live Plot Demo ({duration}s) ===")
    print("Close the plot window to stop...")
    
    # Configure ramp
    driver.set_ramp_frequency(2.0)  # 2 Hz for visible changes
    driver.set_ramp_amplitude(1.0)
    driver.set_ramp_offset(1.5)
    driver.start_ramp()
    
    try:
        # Set up live plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Initialize empty plots
        line_dac, = ax1.plot([], [], 'b-', label='DAC Output', linewidth=2)
        line_adc1, = ax2.plot([], [], 'r-', label='ADC Ch0', linewidth=2)
        line_adc2, = ax2.plot([], [], 'g-', label='ADC Ch1', linewidth=2)
        
        ax1.set_xlim(0, 4000)  # ~1000 samples at 250 MHz ≈ 4ms
        ax1.set_ylim(0, 3)
        ax1.set_ylabel('DAC Voltage (V)')
        ax1.set_title('Live DAC Output')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlim(0, 4000)
        ax2.set_ylim(-2, 2)
        ax2.set_xlabel('Time (μs)')
        ax2.set_ylabel('ADC Voltage (V)')
        ax2.set_title('Live ADC Input (Loopback)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        def update_plot(frame):
            try:
                data = driver.get_paired_data(1000)
                time_us = data['time'] * 1e6
                
                line_dac.set_data(time_us, data['dac'][0])
                line_adc1.set_data(time_us, data['adc'][0])
                line_adc2.set_data(time_us, data['adc'][1])
                
                return line_dac, line_adc1, line_adc2
            except Exception as e:
                print(f"Plot update error: {e}")
                return line_dac, line_adc1, line_adc2
        
        # Create animation
        anim = FuncAnimation(fig, update_plot, interval=100, blit=True, cache_frame_data=False)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Live plot error: {e}")
    finally:
        driver.stop_ramp()
        print("Live plot demo completed")

def run_comprehensive_adc_test():
    """Run comprehensive ADC input testing"""
    print("🔬 CurrentRamp ADC Input Test Suite")
    print("=====================================")
    
    # Connect to device
    try:
        host = os.environ.get('HOST', '192.168.1.20')
        client = connect(host, 'currentramp', restart=False)
        driver = CurrentRamp(client)
        print(f"✅ Connected to {host}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic ADC functionality
    if test_adc_basic_functionality(driver):
        tests_passed += 1
    
    # Test 2: Loopback connection
    loopback_success, loopback_data = test_loopback_connection(driver)
    if loopback_success:
        tests_passed += 1
        
        # Plot the loopback data
        plot_paired_data(loopback_data, "Loopback Test Results")
    
    # Test 3: Live plotting demo
    try:
        live_plot_demo(driver, duration=10)
        tests_passed += 1
    except Exception as e:
        print(f"❌ Live plot demo failed: {e}")
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All ADC tests passed! Your input system is working correctly.")
    elif tests_passed >= 2:
        print("⚠️  Most tests passed. Minor issues may need attention.")
    else:
        print("❌ Several tests failed. Check hardware connections and configuration.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    run_comprehensive_adc_test() 