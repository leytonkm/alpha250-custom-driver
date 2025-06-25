#!/usr/bin/env python3
"""
Live Graph ADC Data
Script to capture and plot ADC data in real-time with interactive controls.
Uses optimized DMA streaming for fast, continuous data acquisition.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
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
    def get_adc_stream_voltages(self, num_samples):
        """Get the most recent ADC voltage samples from the DMA buffer."""
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

class LivePlot:
    def __init__(self, driver, sample_rate_khz, initial_duration_s):
        self.driver = driver
        self.paused = False
        
        # Configure decimation based on sample rate
        self.setup_adc_rate(sample_rate_khz)
        
        self.plot_duration_s = initial_duration_s
        self.plot_points = int(self.plot_duration_s * self.actual_fs)
        print(f"   Fetching {self.plot_points} points per update.")

        # --- Figure and Axes Setup ---
        self.fig = plt.figure(figsize=(14, 8))
        # Main plot area is on the left, widgets on the right
        self.ax = self.fig.add_axes([0.1, 0.1, 0.7, 0.85]) # l, b, w, h
        # Axes for widgets
        self.ax_pause = self.fig.add_axes([0.83, 0.85, 0.15, 0.075])
        self.ax_vscale = self.fig.add_axes([0.83, 0.75, 0.15, 0.075])
        self.ax_vreset = self.fig.add_axes([0.83, 0.65, 0.15, 0.075])
        self.ax_tscale = self.fig.add_axes([0.83, 0.35, 0.15, 0.25])
        
        self.fig.canvas.manager.set_window_title('Live ADC Scope')

        # --- Plotting Objects ---
        self.time_axis = np.arange(self.plot_points) / self.actual_fs
        self.line, = self.ax.plot(self.time_axis, np.zeros(self.plot_points), 'b-', linewidth=1)
        
        self.setup_plot_cosmetics()

        # --- Widgets ---
        self.btn_pause = Button(self.ax_pause, 'Pause')
        self.btn_pause.on_clicked(self.toggle_pause)

        self.btn_vscale = Button(self.ax_vscale, 'Autoscale V')
        self.btn_vscale.on_clicked(self.autoscale_v)

        self.btn_vreset = Button(self.ax_vreset, 'Default V')
        self.btn_vreset.on_clicked(self.default_v)

        self.radio_tscale = RadioButtons(
            self.ax_tscale, 
            ('0.1s', '0.5s', '1.0s', '2.0s', '5.0s'),
            active=2, # Default to 1.0s
            activecolor='blue'
        )
        self.radio_tscale.on_clicked(self.set_time_scale)
        self.ax_tscale.set_title("Time Scale", y=1.0, pad=15)

    def setup_plot_cosmetics(self):
        """Set title, labels, grids etc. for the main plot area."""
        self.ax.set_title("Live ADC Data")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("ADC Voltage (V)")
        self.ax.grid(True, alpha=0.4)
        self.ax.set_xlim(0, self.plot_duration_s)
        self.ax.set_ylim(-0.5, 0.5)

    def setup_adc_rate(self, sample_rate_khz):
        """Configure CIC decimation rate on the device."""
        decimation_rates = {
            30.5: 8192, 50: 5000, 100: 2500, 200: 1250,
        }
        if sample_rate_khz not in decimation_rates:
            print(f"⚠️  Unsupported sample rate {sample_rate_khz}kHz, using 100kHz")
            self.driver.set_cic_decimation_rate(decimation_rates[100])
        else:
            self.driver.set_cic_decimation_rate(decimation_rates[sample_rate_khz])
        
        time.sleep(0.1)
        self.actual_fs = self.driver.get_decimated_sample_rate()
        print(f"   Actual sample rate: {self.actual_fs:.0f} Hz")
    
    def toggle_pause(self, event):
        """Callback to pause/resume the plot."""
        self.paused = not self.paused
        self.btn_pause.label.set_text('Run' if self.paused else 'Pause')

    def autoscale_v(self, event):
        """Callback to autoscale the Y (voltage) axis."""
        if not self.paused:
            print("⚠️  Please pause the plot before autoscaling.")
            return
        y_data = self.line.get_ydata()
        min_v, max_v = np.min(y_data), np.max(y_data)
        margin = (max_v - min_v) * 0.1 + 1e-9 # Add small value to avoid flat lines
        self.ax.set_ylim(min_v - margin, max_v + margin)
    
    def default_v(self, event):
        """Callback to reset the Y axis to the default range."""
        self.ax.set_ylim(-0.5, 0.5)

    def set_time_scale(self, label):
        """Callback to change the time window of the plot."""
        self.plot_duration_s = float(label.replace('s',''))
        self.plot_points = int(self.plot_duration_s * self.actual_fs)
        self.time_axis = np.arange(self.plot_points) / self.actual_fs
        
        # Update plot objects to match new data size
        self.line.set_data(self.time_axis, np.zeros(self.plot_points))
        self.ax.set_xlim(0, self.plot_duration_s)
        print(f"Time scale set to {self.plot_duration_s}s, plotting {self.plot_points} points.")

    def update(self):
        """Fetch new data and update the plot line if not paused."""
        if self.paused:
            return

        voltage_data = self.driver.get_adc_stream_voltages(self.plot_points)
        if len(voltage_data) == self.plot_points:
            self.line.set_ydata(voltage_data)

    def run(self):
        """Start the main plotting loop."""
        # Start streaming on the device
        if not self.driver.is_adc_streaming_active():
            print("   Starting ADC streaming...")
            self.driver.start_adc_streaming()
            time.sleep(1.0) # Let buffer fill a bit
        
        plt.ion() # Interactive mode ON
        print("\n🚀 Live plot started. Close the plot window to stop.")

        while plt.fignum_exists(self.fig.number):
            try:
                self.update()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.05)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️  Error during plotting: {e}")
                time.sleep(1)

        print("\nPlot window closed.")

def main():
    """Main function to set up and run the live plot."""
    print("=" * 60)
    print("🎯 Live ADC Data Graphing Tool")
    print("   Close the plot window to exit.")
    print("=" * 60)
    
    # Connect to device
    driver = connect_to_device()
    if not driver:
        return 1
    
    # === CUSTOMIZE THESE PARAMETERS ===
    sample_rate_khz = 100      # Sample rate: 30.5, 50, 100, or 200 kHz
    initial_duration_s = 1.0   # Initial time window in seconds
    
    print(f"📊 Target sample rate: {sample_rate_khz} kHz")
    
    try:
        plotter = LivePlot(driver, sample_rate_khz, initial_duration_s)
        plotter.run()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        # --- Cleanup ---
        if driver and driver.is_adc_streaming_active():
            driver.stop_adc_streaming()
            print("   ADC streaming stopped.")
        plt.ioff() # Interactive mode OFF
        print("✅ Script finished.")
        return 0

if __name__ == "__main__":
    exit(main()) 