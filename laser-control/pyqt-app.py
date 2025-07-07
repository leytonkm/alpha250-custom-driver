#!/usr/bin/env python3

import sys
import os
import time
import numpy as np
from scipy.signal import find_peaks
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from koheron import connect, command
import math
import logging

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------

DEBUG_ENABLED = os.environ.get('DEBUG', '0') == '1'
logging.basicConfig(
    level=logging.DEBUG if DEBUG_ENABLED else logging.INFO,
    format='[%(asctime)s] %(levelname)s:%(name)s:%(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('LaserControl')

# --- Constants ---
APP_TITLE = "Laser Control - Live ADC Scope"
DEFAULT_HOST = os.environ.get('HOST', '192.168.1.20')
INSTRUMENT_NAME = 'laser-control'

# Performance settings (values will be refined after connecting to instrument)
UPDATE_INTERVAL_MS = 50  # Target plot refresh period (ms)

# The effective number of samples read each update will be computed dynamically
# once we know the decimated sample-rate. 5–10× the descriptor length ensures
# we drain the DMA buffer even if the GUI stalls for one frame.
DEFAULT_SAMPLES_PER_UPDATE = 10000

# Maximum number of visible samples for any capture
MAX_VISIBLE_SAMPLES = 1_000_000  # 1 Mpts
# Safety factor to ensure we over-capture slightly so slow clock doesn't truncate
SAFETY_FACTOR = 1.03  # 3 % head-room

def compute_decimation(run_seconds: float) -> int:
    """Return a CIC decimation rate that keeps visible samples ≤ MAX_VISIBLE_SAMPLES."""
    FS_ADC = 250_000_000  # 250 MHz, defined in config.yml
    target_rate = FS_ADC * run_seconds / MAX_VISIBLE_SAMPLES
    dec_rate = math.ceil(target_rate)
    if dec_rate < 2500:
        dec_rate = 2500  # 100 kS/s upper limit
    return min(dec_rate, 8192)

# --- Driver Interface ---
class CurrentRamp:
    """Python interface for the laser-control driver"""
    def __init__(self, client):
        self.client = client

    @command('CurrentRamp')
    def start_adc_streaming(self): pass
    
    @command('CurrentRamp')
    def stop_adc_streaming(self): pass

    @command('CurrentRamp')
    def get_buffer_position(self):
        return self.client.recv_uint32()

    @command('CurrentRamp')
    def read_adc_buffer_chunk(self, offset, size):
        return self.client.recv_vector(dtype='float32')

    @command('CurrentRamp')
    def read_adc_buffer_block(self, offset, size):
        return self.client.recv_vector(dtype='float32')

    # ---------------------------- Debug helpers ----------------------------
    @command('CurrentRamp')
    def get_current_descriptor_index(self):
        return self.client.recv_uint32()

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
    def get_decimated_sample_rate(self):
        return self.client.recv_double()

    @command('CurrentRamp')
    def set_decimation_rate(self, rate):
        pass

# --- Triggering System ---

class TriggerSystem:
    """Advanced oscilloscope-like triggering system."""
    
    def __init__(self):
        self.mode = 'auto'  # 'auto', 'peak', 'zero_cross', 'edge', 'hysteresis'
        self.level = 0.0
        self.edge = 'rising'  # 'rising', 'falling'
        self.hysteresis = 0.01  # For hysteresis mode
        self.auto_level = True
        self.holdoff_samples = 100  # Minimum samples between triggers
        self.periods_to_display = 2.0  # Default number of periods to show
        
        # Period detection
        self.detected_period = None
        self.period_history = []
        self.max_period_history = 10
        
    def set_trigger_mode(self, mode):
        """Set trigger mode: 'auto', 'peak', 'zero_cross', 'edge', 'hysteresis'"""
        self.mode = mode
        
    def set_trigger_level(self, level):
        """Set trigger level (fraction of signal range)"""
        self.level = level
        self.auto_level = False
        
    def auto_set_level(self, data):
        """Automatically set trigger level to 50% of signal range"""
        if self.auto_level:
            data_min, data_max = np.min(data), np.max(data)
            self.level = (data_min + data_max) / 2
            
    def find_peaks_robust(self, data, min_height_ratio=0.3, min_distance_ratio=0.1):
        """Find peaks with adaptive thresholds"""
        data_range = np.max(data) - np.min(data)
        min_height = np.min(data) + min_height_ratio * data_range
        min_distance = max(10, int(min_distance_ratio * len(data)))
        
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(data, height=min_height, distance=min_distance)
        return peaks, properties
        
    def find_zero_crossings(self, data, threshold=None, edge='rising'):
        """Find zero crossings or threshold crossings"""
        if threshold is None:
            threshold = self.level
            
        if edge == 'rising':
            crossings = np.where((data[:-1] <= threshold) & (data[1:] > threshold))[0]
        elif edge == 'falling':
            crossings = np.where((data[:-1] >= threshold) & (data[1:] < threshold))[0]
        else:  # both
            rising = np.where((data[:-1] <= threshold) & (data[1:] > threshold))[0]
            falling = np.where((data[:-1] >= threshold) & (data[1:] < threshold))[0]
            crossings = np.sort(np.concatenate([rising, falling]))
            
        return crossings + 1  # +1 because we check data[1:]
        
    def hysteresis_trigger(self, data, low_thresh=None, high_thresh=None):
        """Hysteresis triggering to avoid noise"""
        if low_thresh is None:
            data_range = np.max(data) - np.min(data)
            low_thresh = self.level - self.hysteresis * data_range
            high_thresh = self.level + self.hysteresis * data_range
            
        state = 'low'
        triggers = []
        
        for i, sample in enumerate(data):
            if state == 'low' and sample > high_thresh:
                triggers.append(i)
                state = 'high'
            elif state == 'high' and sample < low_thresh:
                state = 'low'
                
        return np.array(triggers)
        
    def detect_period_autocorr(self, data, max_period_ratio=0.8):
        """Detect period using autocorrelation"""
        if len(data) < 100:
            return None
            
        # Remove DC component
        data_centered = data - np.mean(data)
        
        # Autocorrelation
        autocorr = np.correlate(data_centered, data_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Normalize
        autocorr = autocorr / autocorr[0]
        
        # Find peaks, excluding the zero-lag peak
        max_period = int(max_period_ratio * len(data))
        search_range = autocorr[10:max_period]  # Skip first few samples
        
        if len(search_range) < 10:
            return None
            
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(search_range, height=0.3, distance=20)
        
        if len(peaks) > 0:
            period = peaks[0] + 10  # Add back the offset
            return period
            
        return None
        
    def update_period_estimate(self, new_period):
        """Update period estimate with smoothing"""
        if new_period is not None:
            self.period_history.append(new_period)
            if len(self.period_history) > self.max_period_history:
                self.period_history.pop(0)
                
            # Use median for robustness
            self.detected_period = int(np.median(self.period_history))
            
            # Debug logging
            if DEBUG_ENABLED:
                import logging
                logger = logging.getLogger('LaserControl')
                logger.debug(f"Period update: new={new_period}, history={self.period_history}, detected={self.detected_period}")
            
    def find_triggers(self, data):
        """Main trigger detection function"""
        if len(data) < 50:
            return np.array([])
            
        self.auto_set_level(data)
        
        if self.mode == 'auto':
            # Try different methods and pick the best
            triggers = self._auto_trigger(data)
        elif self.mode == 'peak':
            peaks, _ = self.find_peaks_robust(data)
            triggers = peaks
        elif self.mode == 'zero_cross':
            triggers = self.find_zero_crossings(data, edge=self.edge)
        elif self.mode == 'edge':
            # Use derivative for edge detection
            derivative = np.diff(data)
            if self.edge == 'rising':
                from scipy.signal import find_peaks
                edge_points, _ = find_peaks(derivative, distance=self.holdoff_samples)
            else:
                from scipy.signal import find_peaks
                edge_points, _ = find_peaks(-derivative, distance=self.holdoff_samples)
            triggers = edge_points
        elif self.mode == 'hysteresis':
            triggers = self.hysteresis_trigger(data)
        else:
            triggers = np.array([])
            
        # Apply holdoff
        if len(triggers) > 1:
            filtered_triggers = [triggers[0]]
            for trigger in triggers[1:]:
                if trigger - filtered_triggers[-1] >= self.holdoff_samples:
                    filtered_triggers.append(trigger)
            triggers = np.array(filtered_triggers)
            
        # Update period estimate
        if len(triggers) >= 2:
            periods = np.diff(triggers)
            avg_period = np.median(periods)
            self.update_period_estimate(avg_period)
        elif self.mode == 'auto':
            # Try autocorrelation for period detection
            period = self.detect_period_autocorr(data)
            self.update_period_estimate(period)
            
        return triggers
        
    def _auto_trigger(self, data):
        """Automatic trigger mode - tries multiple methods"""
        # Try peak detection first (good for triangular/sawtooth waves)
        peaks, properties = self.find_peaks_robust(data)
        
        if len(peaks) >= 2:
            # Check if peaks are reasonably periodic
            periods = np.diff(peaks)
            period_std = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else float('inf')
            
            if period_std < 0.3:  # Less than 30% variation
                return peaks
        
        # For triangular waves, try finding valleys (negative peaks) as well
        valleys, _ = self.find_peaks_robust(-data)  # Invert signal to find valleys
        
        if len(valleys) >= 2:
            periods = np.diff(valleys)
            period_std = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else float('inf')
            
            if period_std < 0.3:
                return valleys
                
        # Try zero crossing if peaks/valleys aren't good
        crossings = self.find_zero_crossings(data, edge='rising')
        if len(crossings) >= 2:
            periods = np.diff(crossings)
            period_std = np.std(periods) / np.mean(periods) if np.mean(periods) > 0 else float('inf')
            
            if period_std < 0.4:  # Slightly more tolerant for zero crossings
                return crossings
            
        # Try falling edge zero crossings
        crossings = self.find_zero_crossings(data, edge='falling')
        if len(crossings) >= 2:
            return crossings
            
        # Fallback to hysteresis
        return self.hysteresis_trigger(data)
        
    def get_triggered_window(self, data, window_periods=2.0):
        """Get a window of data centered on trigger with specified number of periods"""
        triggers = self.find_triggers(data)
        
        if len(triggers) == 0:
            # No triggers found, return middle portion
            start = len(data) // 4
            end = 3 * len(data) // 4
            return data[start:end], start
            
        # Use the middle trigger if multiple found
        trigger_idx = triggers[len(triggers) // 2]
        
        # Determine window size
        if self.detected_period is not None:
            window_size = int(window_periods * self.detected_period)
        else:
            # Fallback to a reasonable fraction of data
            window_size = len(data) // 2
            
        # Center window on trigger
        start = max(0, trigger_idx - window_size // 2)
        end = min(len(data), trigger_idx + window_size // 2)
        
        return data[start:end], start
        
    def get_status_info(self):
        """Get trigger status information for display"""
        info = {
            'mode': self.mode,
            'level': self.level,
            'edge': self.edge,
            'period': self.detected_period,
            'auto_level': self.auto_level,
            'periods_to_display': self.periods_to_display
        }
        return info

# --- Live Window Worker (infinite rolling oscilloscope) ---

class LiveWorker(QtCore.QThread):
    """Continuously fetch the last `window_seconds` of data each refresh."""
    window_ready = QtCore.pyqtSignal(np.ndarray)
    status_changed = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, driver, sample_rate, window_seconds, guard_pts, refresh_ms=UPDATE_INTERVAL_MS):
        super().__init__()
        self.driver = driver
        self.sample_rate = sample_rate
        self.window_pts = int(window_seconds * sample_rate)
        self.guard_pts = guard_pts
        self.refresh_ms = refresh_ms

        # DMA ring params – keep in sync with firmware
        self.n_desc = 512
        self.n_pts = 2048
        self.total_buffer_size = self.n_desc * self.n_pts

        self.running = False
        self.debug_counter = 0  # for periodic prints

    def run(self):
        try:
            self.driver.start_adc_streaming()
            self.running = True
            self.status_changed.emit("LiveWorker started")

            # Initialize read pointer safe distance behind writer
            write_ptr = self.driver.get_buffer_position()
            self.safe_lag_desc = 8  # same as GUI
            self.read_ptr = (write_ptr - self.safe_lag_desc * self.n_pts + self.total_buffer_size) % self.total_buffer_size

            while self.running:
                loop_start = time.time()

                write_ptr = self.driver.get_buffer_position()
                samples_available = (write_ptr - self.read_ptr + self.total_buffer_size) % self.total_buffer_size

                # ensure we don't catch descriptor being written
                safe_margin = self.safe_lag_desc * self.n_pts
                if samples_available <= safe_margin:
                    samples_available = 0
                else:
                    samples_available -= safe_margin

                if samples_available > 0:
                    read_size = min(samples_available, DEFAULT_SAMPLES_PER_UPDATE)
                    data = np.array(
                        self.driver.read_adc_buffer_block(self.read_ptr, read_size), dtype=np.float32)
                    if data.size > 0:
                        self.window_ready.emit(data)
                        self.read_ptr = (self.read_ptr + data.size) % self.total_buffer_size

                elapsed = (time.time() - loop_start) * 1000

                # ---------------- Debug instrumentation ----------------
                if DEBUG_ENABLED:
                    self.debug_counter += 1
                    if self.debug_counter >= 5:  # Every ~250ms (5 × 50 ms) for faster idle detection
                        self.debug_counter = 0
                        desc = self.driver.get_current_descriptor_index()
                        dma_run = self.driver.get_dma_running()
                        dma_idle = self.driver.get_dma_idle()
                        dma_err = self.driver.get_dma_error()
                        logger.debug(
                            "LiveWorker RD=%d WR=%d avail=%d desc=%d running=%s idle=%s err=%s",
                            self.read_ptr, write_ptr, samples_available, desc, dma_run, dma_idle, dma_err)
                        # Ultra-fast restart for minimal gaps
                        if dma_idle and not dma_err:
                            logger.debug("DMA idle detected - ultra-fast restart")
                            self.driver.stop_adc_streaming()
                            time.sleep(0.0001)  # Just 100µs delay
                            self.driver.start_adc_streaming()
                            # Re-align read pointer
                            write_ptr = self.driver.get_buffer_position()
                            self.read_ptr = (write_ptr - self.safe_lag_desc * self.n_pts + self.total_buffer_size) % self.total_buffer_size
                            continue
                        if dma_err:
                            logger.warning("DMA error detected - attempting restart")
                            self.driver.stop_adc_streaming()
                            time.sleep(0.001)  # 1ms delay
                            self.driver.start_adc_streaming()
                            # Re-align read pointer
                            write_ptr = self.driver.get_buffer_position()
                            self.read_ptr = (write_ptr - self.safe_lag_desc * self.n_pts + self.total_buffer_size) % self.total_buffer_size
                            continue

                time.sleep(max(0, self.refresh_ms - elapsed) / 1000)

        except Exception as e:
            self.status_changed.emit(f"LiveWorker error: {e}")
        finally:
            try:
                self.driver.stop_adc_streaming()
            except Exception:
                pass
            self.status_changed.emit("LiveWorker stopped")
            self.finished.emit()

    def stop(self):
        self.running = False
        # The finally block in run() will handle stopping the stream

# --- Data Acquisition Thread ---

class RunWorker(QtCore.QThread):
    """
    Worker thread for acquiring ADC data from the instrument.
    Decouples data acquisition from the GUI to prevent freezing.
    Can run indefinitely (live stream) or for a fixed number of samples.
    """
    data_ready = QtCore.pyqtSignal(np.ndarray)
    status_changed = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    progress_updated = QtCore.pyqtSignal(int) # For fixed-duration runs

    def __init__(self, driver, samples_per_update=DEFAULT_SAMPLES_PER_UPDATE, samples_to_collect=None):
        super(RunWorker, self).__init__()
        self.driver = driver
        self.running = False
        # DMA buffer configuration – keep in sync with firmware & C++ driver
        self.n_desc = 512  # Matches C++ driver
        self.n_pts  = 2048  # 512 × 64-bit words per descriptor → 20.48 ms @ 100 kS/s
        self.total_buffer_size = self.n_desc * self.n_pts
        self.samples_per_update = samples_per_update
        # Initialise read pointer a few descriptors behind the current write pointer
        self.safe_lag_desc = 8  # stay several descriptors behind write head for safety (≈16 ms @ 100 kS/s)
        self.read_ptr = 0  # will be set after warm-up
        self.samples_to_collect = samples_to_collect
        self.samples_collected = 0
        self.debug_counter = 0

    def run(self):
        """Main acquisition loop."""
        try:
            self.status_changed.emit("Worker thread started")
            self.running = True

            self.driver.start_adc_streaming()
            
            # --- Robust warm-up: Actively read and discard initial data ---
            self.status_changed.emit("ADC warming up...")
            warmup_end_time = time.time() + 0.5
            while time.time() < warmup_end_time:
                write_ptr = self.driver.get_buffer_position()
                samples_available = (write_ptr - self.read_ptr + self.total_buffer_size) % self.total_buffer_size
                if samples_available > 0:
                    read_size = min(samples_available, self.samples_per_update)
                    # Read data to advance our pointer, but do nothing with the returned data
                    _ = self.driver.read_adc_buffer_block(self.read_ptr, read_size)
                    self.read_ptr = (self.read_ptr + read_size) % self.total_buffer_size
                time.sleep(0.02) # Don't hog the CPU while waiting
            
            # After warm-up, synchronise read pointer safe distance behind current write pointer
            write_ptr = self.driver.get_buffer_position()
            self.read_ptr = (write_ptr - self.safe_lag_desc * self.n_pts + self.total_buffer_size) % self.total_buffer_size
            self.status_changed.emit(f"Starting acquisition at sample index {self.read_ptr}")

            while self.running:
                loop_start_time = time.time()
                
                write_ptr = self.driver.get_buffer_position()
                samples_available = (write_ptr - self.read_ptr + self.total_buffer_size) % self.total_buffer_size

                # Keep a safety margin so we never overrun the DMA writer.
                safe_margin_samples = self.safe_lag_desc * self.n_pts
                if samples_available <= safe_margin_samples:
                    samples_available = 0
                else:
                    samples_available -= safe_margin_samples

                if samples_available > 0:
                    read_size = min(samples_available, self.samples_per_update)
                    
                    if self.samples_to_collect is not None:
                        remaining = self.samples_to_collect - self.samples_collected
                        if remaining <= 0:
                            self.running = False
                            continue
                        read_size = min(read_size, remaining)

                    if read_size > 0:
                        data = np.array(self.driver.read_adc_buffer_block(self.read_ptr, read_size), dtype=np.float32)
                        
                        if data.size > 0:
                            self.data_ready.emit(data)
                            self.read_ptr = (self.read_ptr + data.size) % self.total_buffer_size
                            self.samples_collected += data.size
                            
                            if self.samples_to_collect is not None:
                                progress = int(100 * self.samples_collected / self.samples_to_collect)
                                self.progress_updated.emit(progress)

                elapsed_ms = (time.time() - loop_start_time) * 1000

                if DEBUG_ENABLED:
                    self.debug_counter += 1
                    if self.debug_counter >= 20:
                        self.debug_counter = 0
                        desc = self.driver.get_current_descriptor_index()
                        logger.debug(
                            "RunWorker RD=%d WR=%d avail=%d collected=%d/%s desc=%d",
                            self.read_ptr, write_ptr, samples_available, self.samples_collected,
                            self.samples_to_collect if self.samples_to_collect is not None else '∞', desc)

                sleep_ms = max(0, UPDATE_INTERVAL_MS - elapsed_ms)
                time.sleep(sleep_ms / 1000)

        except Exception as e:
            self.status_changed.emit(f"Error in worker: {e}")
        finally:
            # Always stop ADC streaming at the end of a worker run to leave the hardware in a
            # defined state for the next acquisition.
            try:
                self.driver.stop_adc_streaming()
            except Exception:
                pass
            self.status_changed.emit("ADC streaming stopped.")
            self.finished.emit()

    def stop(self):
        self.running = False
        # The finally block in run() will handle stopping the stream

# --- Main Application Window ---

class MainWindow(QtWidgets.QMainWindow):
    """Main GUI window."""
    MAX_POINTS_TO_PLOT = 10000 # Performance: max points to send to setData at once

    def __init__(self, driver):
        super(MainWindow, self).__init__()
        self.driver = driver
        self.run_worker = None
        self.live_worker = None

        # --- Data Buffers ---
        self.time_scale_s = 5.0 # Default time window to display
        self.sample_rate = self.driver.get_decimated_sample_rate()
        
        # --- Trigger System ---
        self.trigger_system = TriggerSystem()
        self.trigger_enabled = False
        
        # Allocate live buffers sized to current time window plus safety margin
        self.resize_live_buffers()
        self.buffer_ptr = 0
        self.sample_clock = 0  # 64-bit counter of total samples received
        self.last_time = 0.0
        self.total_samples_received = 0
        self.is_fixed_run = False
        self.run_duration = 0.0
        self.crosshair_locked = False

        self.setup_ui()
        self.connect_signals()

        self.set_time_scale(self.time_scale_s)
        self.status_bar.showMessage("Ready. Connect to an instrument and start streaming.")

    # ------------------------------------------------------------------
    # Dynamic buffer allocation for rolling oscilloscope view
    # ------------------------------------------------------------------
    def resize_live_buffers(self):
        """Allocate circular buffers sized for the current time window."""
        # Visible points in the window
        visible_pts = int(self.time_scale_s * self.sample_rate)
        # Add guard margin (8 DMA descriptors)
        guard_pts = 8 * 2048
        total_pts = visible_pts + guard_pts

        self.max_plot_points = total_pts
        self.time_buffer = np.zeros(total_pts, dtype=np.float32)
        self.voltage_buffer = np.zeros(total_pts, dtype=np.float32)

    def setup_ui(self):
        """Initialize UI elements."""
        self.setWindowTitle(APP_TITLE)
        self.resize(1200, 800)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)

        # --- Control Panel ---
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_panel.setFixedWidth(250)

        # --- Live Streaming Controls ---
        streaming_group_box = QtWidgets.QGroupBox("Live View")
        streaming_layout = QtWidgets.QVBoxLayout()
        self.start_button = QtWidgets.QPushButton("Enable Live View")
        self.start_button.setCheckable(True)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #90EE90;
                border-color: #4CAF50;
            }
        """)
        self.ts_group_box = QtWidgets.QGroupBox("Time Scale")
        ts_layout = QtWidgets.QVBoxLayout()
        self.ts_radios = {}
        for val in [1.0, 2.0, 5.0, 10.0]:
            self.ts_radios[val] = QtWidgets.QRadioButton(f"{val} s")
            ts_layout.addWidget(self.ts_radios[val])
        self.ts_radios[self.time_scale_s].setChecked(True)
        self.ts_group_box.setLayout(ts_layout)
        streaming_layout.addWidget(self.start_button)
        streaming_layout.addWidget(self.ts_group_box)
        streaming_group_box.setLayout(streaming_layout)

        # --- Trigger Controls ---
        trigger_group_box = QtWidgets.QGroupBox("Trigger")
        trigger_layout = QtWidgets.QVBoxLayout()
        
        # Trigger enable toggle button
        self.trigger_enable_button = QtWidgets.QPushButton("Enable Trigger")
        self.trigger_enable_button.setCheckable(True)
        self.trigger_enable_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #FFE4B5;
                border-color: #FFA500;
            }
        """)
        trigger_layout.addWidget(self.trigger_enable_button)
        
        # Freeze toggle button (moved here for better workflow)
        self.freeze_button = QtWidgets.QPushButton("Freeze Display")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #FFB6C1;
                border-color: #FF69B4;
            }
        """)
        trigger_layout.addWidget(self.freeze_button)
        
        # Trigger mode selection
        trigger_mode_layout = QtWidgets.QHBoxLayout()
        trigger_mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.trigger_mode_combo = QtWidgets.QComboBox()
        self.trigger_mode_combo.addItems(['Auto', 'Peak', 'Zero Cross', 'Edge', 'Hysteresis'])
        self.trigger_mode_combo.setCurrentText('Auto')
        trigger_mode_layout.addWidget(self.trigger_mode_combo)
        trigger_layout.addLayout(trigger_mode_layout)
        
        # Trigger level control
        trigger_level_layout = QtWidgets.QHBoxLayout()
        trigger_level_layout.addWidget(QtWidgets.QLabel("Level:"))
        self.trigger_level_spinbox = QtWidgets.QDoubleSpinBox()
        self.trigger_level_spinbox.setRange(-10.0, 10.0)
        self.trigger_level_spinbox.setDecimals(3)
        self.trigger_level_spinbox.setSingleStep(0.001)
        self.trigger_level_spinbox.setSuffix(" V")
        self.trigger_level_spinbox.setValue(0.0)
        trigger_level_layout.addWidget(self.trigger_level_spinbox)
        trigger_layout.addLayout(trigger_level_layout)
        
        # Trigger edge selection
        trigger_edge_layout = QtWidgets.QHBoxLayout()
        trigger_edge_layout.addWidget(QtWidgets.QLabel("Edge:"))
        self.trigger_edge_combo = QtWidgets.QComboBox()
        self.trigger_edge_combo.addItems(['Rising', 'Falling'])
        trigger_edge_layout.addWidget(self.trigger_edge_combo)
        trigger_layout.addLayout(trigger_edge_layout)
        
        # Auto level button
        self.auto_level_button = QtWidgets.QPushButton("Auto Level")
        trigger_layout.addWidget(self.auto_level_button)
        
        # Periods to display control
        periods_layout = QtWidgets.QHBoxLayout()
        periods_layout.addWidget(QtWidgets.QLabel("Periods:"))
        self.periods_spinbox = QtWidgets.QDoubleSpinBox()
        self.periods_spinbox.setRange(1.0, 5.0)
        self.periods_spinbox.setDecimals(1)
        self.periods_spinbox.setSingleStep(0.5)
        self.periods_spinbox.setValue(2.0)  # Default to 2 periods
        periods_layout.addWidget(self.periods_spinbox)
        trigger_layout.addLayout(periods_layout)
        
        # Trigger status display
        self.trigger_status_label = QtWidgets.QLabel("Status: Disabled")
        self.trigger_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.trigger_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        trigger_layout.addWidget(self.trigger_status_label)
        
        trigger_group_box.setLayout(trigger_layout)

        # --- Fixed Run Controls ---
        run_group_box = QtWidgets.QGroupBox("Fixed-Duration Run")
        run_layout = QtWidgets.QGridLayout()
        # Duration control
        run_layout.addWidget(QtWidgets.QLabel("Duration:"), 0, 0)
        self.duration_spinbox = QtWidgets.QDoubleSpinBox()
        self.duration_spinbox.setRange(0.0001, 60.0)
        self.duration_spinbox.setDecimals(3)
        self.duration_spinbox.setSingleStep(0.1)
        self.duration_spinbox.setSuffix(" s")
        self.duration_spinbox.setValue(1.0)
        run_layout.addWidget(self.duration_spinbox, 0, 1)

        # Sample rate control (visible rate after decimation)
        run_layout.addWidget(QtWidgets.QLabel("Rate:"), 1, 0)
        self.rate_spinbox = QtWidgets.QSpinBox()
        self.rate_spinbox.setRange(1, 1000)  # 1 kS/s to 1 MS/s
        self.rate_spinbox.setSuffix(" kHz")
        self.rate_spinbox.setValue(100)  # default 100 kS/s
        run_layout.addWidget(self.rate_spinbox, 1, 1)
        self.run_button = QtWidgets.QPushButton("Start Run")
        run_layout.addWidget(self.run_button, 2, 0, 1, 2)
        self.effective_rate_label = QtWidgets.QLabel("Rate: -- kHz")
        self.effective_rate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        run_layout.addWidget(self.effective_rate_label, 3, 0, 1, 2)
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setTextVisible(False)
        run_layout.addWidget(self.run_progress_bar, 4, 0, 1, 2)
        run_group_box.setLayout(run_layout)

        # --- Locked Coordinate Display ---
        locked_coord_group = QtWidgets.QGroupBox("Locked Coordinate")
        locked_coord_layout = QtWidgets.QVBoxLayout()
        self.locked_coord_label = QtWidgets.QLabel("Time: --\nVoltage: --")
        self.locked_coord_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        locked_coord_layout.addWidget(self.locked_coord_label)
        locked_coord_group.setLayout(locked_coord_layout)

        control_layout.addWidget(streaming_group_box)
        control_layout.addWidget(trigger_group_box)
        control_layout.addWidget(run_group_box)
        control_layout.addWidget(locked_coord_group)
        # --- Live Statistics Display ---
        stats_group = QtWidgets.QGroupBox("Live Stats (visible window)")
        stats_layout = QtWidgets.QVBoxLayout()
        self.stats_label = QtWidgets.QLabel("min: -- V\nmax: -- V\nRMS: -- V\nP2P: -- V")
        self.stats_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        stats_layout.addWidget(self.stats_label)
        stats_group.setLayout(stats_layout)

        control_layout.addStretch()
        control_layout.addWidget(stats_group)

        # --- Plotting Areas ---
        plots_container = QtWidgets.QVBoxLayout()

        # Live plot (top)
        self.plot_widget = pg.PlotWidget()
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        self.plot_curve.setDownsampling(auto=True)
        self.plot_widget.setClipToView(True)
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.plot_widget.setLabel('left', "Voltage", units='V')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(-0.5, 0.5)
        self.plot_widget.setXRange(0, 1.0)  # Set initial X range to avoid overflow warnings

        # --- Desmos-style Hover Elements (live plot only) ---
        self.hover_marker = pg.ScatterPlotItem([], [], pxMode=True, size=10, pen=pg.mkPen('w'), brush=pg.mkBrush(255, 255, 255, 120))
        self.locked_marker = pg.ScatterPlotItem([], [], pxMode=True, size=12, pen=pg.mkPen('w'), brush=pg.mkBrush('b'))
        self.coord_text = pg.TextItem(text='', color=(200, 200, 200), anchor=(-0.1, 1.2))
        
        # --- Trigger Visualization ---
        self.trigger_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=2, style=QtCore.Qt.PenStyle.DashLine))
        self.trigger_line.setVisible(False)
        self.trigger_markers = pg.ScatterPlotItem([], [], pxMode=True, size=8, pen=pg.mkPen('r'), brush=pg.mkBrush('r'))
        
        self.plot_widget.addItem(self.hover_marker)
        self.plot_widget.addItem(self.locked_marker)
        self.plot_widget.addItem(self.coord_text)
        self.plot_widget.addItem(self.trigger_line)
        self.plot_widget.addItem(self.trigger_markers)
        
        plots_container.addWidget(self.plot_widget)

        # Run plot (bottom) hidden until a fixed run completes
        self.run_plot_widget = pg.PlotWidget()
        self.run_plot_curve = self.run_plot_widget.plot(pen=pg.mkPen('g', width=2))
        self.run_plot_widget.setLabel('bottom', "Run Time", units='s')
        self.run_plot_widget.setLabel('left', "Voltage", units='V')
        self.run_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.run_plot_widget.setVisible(False)

        plots_container.addWidget(self.run_plot_widget)

        main_layout.addLayout(plots_container)
        main_layout.addWidget(control_panel)

        # --- Status Bar ---
        self.status_bar = self.statusBar()

    def connect_signals(self):
        """Connect UI signals to slots."""
        self.start_button.toggled.connect(self.toggle_streaming)
        self.run_button.clicked.connect(self.start_fixed_run)
        for val, radio in self.ts_radios.items():
            radio.toggled.connect(lambda checked, v=val: self.set_time_scale(v) if checked else None)

        # Update max duration label dynamically
        self.duration_spinbox.valueChanged.connect(self.update_max_duration)
        self.rate_spinbox.valueChanged.connect(self.update_max_duration)

        # Connect plot interaction signals
        self.plot_widget.scene().sigMouseMoved.connect(self.update_crosshair)
        self.plot_widget.scene().sigMouseClicked.connect(self.plot_clicked)

        # Connect trigger controls
        self.trigger_enable_button.toggled.connect(self.on_trigger_enable_changed)
        self.trigger_mode_combo.currentTextChanged.connect(self.on_trigger_mode_changed)
        self.trigger_level_spinbox.valueChanged.connect(self.on_trigger_level_changed)
        self.trigger_edge_combo.currentTextChanged.connect(self.on_trigger_edge_changed)
        self.auto_level_button.clicked.connect(self.on_auto_level_clicked)
        self.periods_spinbox.valueChanged.connect(self.on_periods_changed)

        # initialise max-duration label
        QtCore.QTimer.singleShot(0, self.update_max_duration)

    def update_crosshair(self, pos):
        """Handle mouse movement on the plot for crosshair."""
        if self.crosshair_locked:
            return

        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
        x_data, y_data = self.plot_curve.getData()
        
        if x_data is None or len(x_data) == 0:
            return
        
        # --- High-Precision Point Selection using Normalized Euclidean Distance ---
        # Get the visible range of the plot to normalize the coordinates
        view_range = self.plot_widget.getPlotItem().vb.viewRange()
        x_span = view_range[0][1] - view_range[0][0]
        y_span = view_range[1][1] - view_range[1][0]
        
        # Avoid division by zero if the view is not yet set
        if x_span == 0 or y_span == 0:
            return

        # Calculate the normalized distance squared from the mouse to every point
        dx = (x_data - mouse_point.x()) / x_span
        dy = (y_data - mouse_point.y()) / y_span
        dist_sq = dx**2 + dy**2
        
        # Find the index of the point with the minimum distance
        index = np.argmin(dist_sq)
        snap_x, snap_y = x_data[index], y_data[index]
        
        # Convert the snapped data point back to scene/pixel coordinates for distance check
        snap_point_in_scene = self.plot_widget.getPlotItem().vb.mapViewToScene(pg.Point(snap_x, snap_y))

        # Calculate the distance in pixels (scene coordinates)
        distance = (snap_point_in_scene - pos).manhattanLength()
        
        # Only show the hover elements if the mouse is close to the line
        if distance < 30: # 30-pixel threshold
            self.hover_marker.setData([snap_x], [snap_y])
            self.coord_text.setText(f"({snap_x:.4f} s, {snap_y:.3f} V)")
            self.coord_text.setPos(snap_x, snap_y)
        else:
            self.hover_marker.setData([], [])
            self.coord_text.setText('')

    def plot_clicked(self, event):
        self.crosshair_locked = not self.crosshair_locked
        
        x_data, y_data = self.plot_curve.getData()
        if x_data is None or len(x_data) == 0:
            self.crosshair_locked = False # Can't lock if there's no data
            return

        if self.crosshair_locked:
            # On lock, hide hover elements and show locked elements
            self.hover_marker.setData([], [])
            self.coord_text.setText('')

            mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(event.scenePos())
            
            # Use the same high-precision logic for locking the point
            view_range = self.plot_widget.getPlotItem().vb.viewRange()
            x_span = view_range[0][1] - view_range[0][0]
            y_span = view_range[1][1] - view_range[1][0]
            
            if x_span == 0 or y_span == 0:
                self.crosshair_locked = False
                return

            dx = (x_data - mouse_point.x()) / x_span
            dy = (y_data - mouse_point.y()) / y_span
            dist_sq = dx**2 + dy**2
            
            index = np.argmin(dist_sq)
            lock_x, lock_y = x_data[index], y_data[index]
            
            self.locked_marker.setData([lock_x], [lock_y])
            self.locked_coord_label.setText(f"Time: {lock_x:.4f} s\nVoltage: {lock_y:.3f} V")
        else:
            # On unlock, clear the locked elements
            self.locked_marker.setData([], [])
            self.locked_coord_label.setText("Time: --\nVoltage: --")

    # --- Trigger Control Methods ---
    
    def on_trigger_enable_changed(self, enabled):
        """Handle trigger enable/disable"""
        self.trigger_enabled = enabled
        self.trigger_line.setVisible(enabled)
        
        if enabled:
            # Auto-start live view when trigger is enabled
            if not self.start_button.isChecked():
                self.start_button.setChecked(True)
                self.start_streaming()
            
            self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()}")
            self.trigger_status_label.setStyleSheet("QLabel { background-color: #90EE90; padding: 5px; }")
        else:
            self.trigger_status_label.setText("Status: Disabled")
            self.trigger_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
            self.trigger_markers.setData([], [])
            
    def on_trigger_mode_changed(self, mode_text):
        """Handle trigger mode change"""
        mode_map = {
            'Auto': 'auto',
            'Peak': 'peak', 
            'Zero Cross': 'zero_cross',
            'Edge': 'edge',
            'Hysteresis': 'hysteresis'
        }
        self.trigger_system.set_trigger_mode(mode_map[mode_text])
        
        if self.trigger_enabled:
            self.trigger_status_label.setText(f"Status: {mode_text}")
            
    def on_trigger_level_changed(self, level):
        """Handle trigger level change"""
        self.trigger_system.set_trigger_level(level)
        self.trigger_line.setPos(level)
        
    def on_trigger_edge_changed(self, edge_text):
        """Handle trigger edge change"""
        self.trigger_system.edge = edge_text.lower()
        
    def on_auto_level_clicked(self):
        """Auto-set trigger level based on current data"""
        if hasattr(self, 'voltage_buffer') and self.total_samples_received > 0:
            # Use recent data for auto-level
            recent_samples = min(1000, self.total_samples_received)
            end_ptr = self.buffer_ptr
            start_ptr = (end_ptr - recent_samples + self.max_plot_points) % self.max_plot_points
            
            if start_ptr < end_ptr:
                recent_data = self.voltage_buffer[start_ptr:end_ptr]
            else:
                recent_data = np.concatenate((self.voltage_buffer[start_ptr:], self.voltage_buffer[:end_ptr]))
                
            if len(recent_data) > 0:
                auto_level = np.mean([np.min(recent_data), np.max(recent_data)])
                self.trigger_level_spinbox.setValue(auto_level)
                self.trigger_system.auto_level = True

    def on_periods_changed(self):
        """Update the number of periods to display in the triggered window."""
        self.trigger_system.periods_to_display = self.periods_spinbox.value()

    @QtCore.pyqtSlot(bool)
    def toggle_streaming(self, checked):
        if checked:
            self.start_streaming()
            self.start_button.setText("Stop Streaming")
        else:
            self.stop_streaming()
            self.start_button.setText("Start Streaming")

    def start_streaming(self):
        """Start the live acquisition worker."""
        self.is_fixed_run = False
        self.run_button.setEnabled(False)  # Disable run button
        self.reset_and_clear_buffers()
        
        # Clear any existing DMA errors before starting
        try:
            self.driver.stop_adc_streaming()
            time.sleep(0.001)  # Brief pause to ensure clean state
        except:
            pass  # Ignore errors when stopping (might not be running)
        
        dec_rate = compute_decimation(self.time_scale_s)
        self.driver.set_decimation_rate(dec_rate)
        self.sample_rate = self.driver.get_decimated_sample_rate()
        
        self.resize_live_buffers() # Ensure buffers are sized for this view

        guard_pts = 8 * 2048
        self.live_worker = LiveWorker(self.driver, self.sample_rate, self.time_scale_s, guard_pts)
        self.live_worker.window_ready.connect(self.update_live_plot)
        self.live_worker.status_changed.connect(self.status_bar.showMessage)
        self.live_worker.finished.connect(self.on_live_worker_finished)
        self.live_worker.start()

    def stop_streaming(self):
        """Stop the live acquisition worker."""
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker.wait() # Wait for thread to finish cleanly

    def start_fixed_run(self):
        """Start the data acquisition worker for a fixed duration."""
        # Compute decimation from user-selected sample rate
        desired_rate_k = self.rate_spinbox.value()
        desired_rate = desired_rate_k * 1000
        # Clamp to allowed range
        if desired_rate <= 0:
            desired_rate = 100000
        dec_rate = math.ceil(250_000_000 / desired_rate)
        dec_rate = max(10, min(8192, dec_rate))

        # Check that requested duration fits in max points
        max_duration = MAX_VISIBLE_SAMPLES / desired_rate
        if self.run_duration > max_duration:
            QtWidgets.QMessageBox.warning(self, "Duration too long", 
                f"At {desired_rate_k} kHz the maximum duration is {max_duration:.2f} s.")
            return

        self.driver.set_decimation_rate(dec_rate)
        self.sample_rate = self.driver.get_decimated_sample_rate()
        self.effective_rate_label.setText(f"Rate: {self.sample_rate/1e3:.1f} kHz  |  Max {max_duration:.2f} s")
        self.is_fixed_run = True
        self.run_duration = self.duration_spinbox.value()
        visible_samples_goal = math.ceil(self.run_duration * self.sample_rate * SAFETY_FACTOR)
        self.target_visible_samples = math.ceil(self.run_duration * self.sample_rate)  # exact requested
        guard = 8 * 2048  # safe lag (8 descriptors) * samples per descriptor
        self.current_guard = guard  # remember for post-processing
        samples_to_collect = visible_samples_goal + guard
        
        self.start_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.duration_spinbox.setEnabled(False)
        self.rate_spinbox.setEnabled(False)
        self.ts_group_box.setEnabled(False)
        
        self.plot_widget.setXRange(0, self.run_duration)
        self.run_progress_bar.setValue(0)
        
        self.reset_and_clear_buffers()
        
        samples_per_update = max(DEFAULT_SAMPLES_PER_UPDATE, int(0.1 * self.sample_rate))
        self.run_worker = RunWorker(self.driver, samples_per_update=samples_per_update, samples_to_collect=samples_to_collect)
        self.connect_run_worker_signals()
        self.run_worker.progress_updated.connect(self.run_progress_bar.setValue)
        self.run_worker.start()

    def reset_and_clear_buffers(self):
        self.buffer_ptr = 0
        self.sample_clock = 0  # Reset global sample counter
        self.last_time = 0.0
        self.total_samples_received = 0
        self.voltage_buffer.fill(0)
        self.time_buffer.fill(0)
        self.plot_curve.setData(x=[], y=[])
        
        # Reset markers and labels
        self.crosshair_locked = False
        self.hover_marker.setData([], [])
        self.locked_marker.setData([], [])
        self.coord_text.setText('')
        self.locked_coord_label.setText("Time: --\nVoltage: --")

    def _get_decimated_window(self):
        """Return (x, y) arrays decimated to <= MAX_POINTS_TO_PLOT covering the visible window."""
        points_to_show = int(self.time_scale_s * self.sample_rate)
        available = min(points_to_show, self.total_samples_received)
        if available == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

        end_ptr = self.buffer_ptr
        start_ptr = (end_ptr - available + self.max_plot_points) % self.max_plot_points

        num_points = min(self.MAX_POINTS_TO_PLOT, available)
        # Evenly spaced indices within the window
        if num_points == available:
            # Rare case where window smaller than MAX_POINTS_TO_PLOT
            if start_ptr < end_ptr:
                x_data = self.time_buffer[start_ptr:end_ptr]
                y_data = self.voltage_buffer[start_ptr:end_ptr]
            else:
                x_data = np.concatenate((self.time_buffer[start_ptr:], self.time_buffer[:end_ptr]))
                y_data = np.concatenate((self.voltage_buffer[start_ptr:], self.voltage_buffer[:end_ptr]))
        else:
            # Use linspace to avoid allocating large arrays
            idx_offsets = (np.linspace(0, available - 1, num_points)).astype(np.int64)
            indices = (start_ptr + idx_offsets) % self.max_plot_points
            x_data = self.time_buffer[indices]
            y_data = self.voltage_buffer[indices]
        # Shift to window [0, time_scale_s]
        window_end_time = self.sample_clock / self.sample_rate
        window_start_time = max(0.0, window_end_time - self.time_scale_s)
        x_data = x_data - window_start_time
        return x_data, y_data

    def connect_run_worker_signals(self):
        self.run_worker.data_ready.connect(self.update_run_plot)
        self.run_worker.status_changed.connect(self.status_bar.showMessage)
        self.run_worker.finished.connect(self.on_run_worker_finished)

    def on_run_worker_finished(self):
        """Called when a fixed duration run completes."""
        self.run_worker = None
        self.start_button.setEnabled(True)
        self.start_button.setChecked(False)
        self.run_button.setEnabled(True)
        self.duration_spinbox.setEnabled(True)
        self.rate_spinbox.setEnabled(True)
        self.ts_group_box.setEnabled(True)

        if self.is_fixed_run:
            # Remove guard samples from analysis
            guard = getattr(self, 'current_guard', 0)
            target = getattr(self, 'target_visible_samples', None)
            visible_samples = max(0, self.total_samples_received - guard)
            if target is not None and visible_samples > target:
                visible_samples = target  # trim overshoot

            # Re-calibrate sample rate from actual number of visible samples
            if self.run_duration > 0 and visible_samples > 0:
                new_rate = visible_samples / self.run_duration
                # Update stored sample_rate if it changed noticeably
                if abs(new_rate - self.sample_rate) / self.sample_rate > 0.01:
                    self.sample_rate = new_rate
                # Always refresh the final plot with precisely trimmed data
                x_data = np.arange(visible_samples) / self.sample_rate
                y_data = self.voltage_buffer[:visible_samples]
                self.plot_curve.setData(x=x_data, y=y_data)
                # update duration in case of slight mismatch
                run_duration_actual = visible_samples / self.sample_rate
                self.plot_widget.setXRange(0, run_duration_actual, padding=0.05)
            else:
                self.plot_curve.setData(x=[], y=[])

            self.status_bar.showMessage(f"Run of {self.run_duration}s finished. {visible_samples} samples collected.")
            self.run_progress_bar.setValue(100)

            # Show run plot panel and plot data
            self.run_plot_curve.setData(x=x_data, y=y_data)
            self.run_plot_widget.setVisible(True)
        else:
             self.status_bar.showMessage("Streaming stopped.")
    
    def on_live_worker_finished(self):
        """Called when live streaming is stopped."""
        self.live_worker = None
        self.run_button.setEnabled(True) # Re-enable run button
        self.start_button.setChecked(False)
        self.status_bar.showMessage("Live view stopped.")

    @QtCore.pyqtSlot(float)
    def set_time_scale(self, scale_s):
        """Update the visible time window."""
        self.time_scale_s = scale_s
        # Update decimation only if in live-streaming mode (worker running & not fixed run)
        if self.run_worker and not self.is_fixed_run:
            dec_rate = compute_decimation(self.time_scale_s)
            self.driver.set_decimation_rate(dec_rate)
            self.sample_rate = self.driver.get_decimated_sample_rate()
        # Reallocate live buffers for new window and clear data
        self.resize_live_buffers()
        self.reset_and_clear_buffers()
        self.plot_curve.setData(x=[], y=[])
        self.status_bar.showMessage(f"Time scale set to {scale_s}s (rate {self.sample_rate/1e3:.1f} kHz)")

    @QtCore.pyqtSlot(np.ndarray)
    def update_run_plot(self, new_data):
        """Append new data during fixed-run and update the plot efficiently."""
        if self.freeze_button.isChecked():
            return
        n_new = new_data.size
        if n_new == 0:
            return
        self.sample_clock += n_new
        new_times = (self.sample_clock / self.sample_rate) - (np.arange(n_new, 0, -1) / self.sample_rate)
        start_idx = self.buffer_ptr
        indices = np.arange(start_idx, start_idx + n_new) % self.max_plot_points
        self.voltage_buffer[indices] = new_data
        self.time_buffer[indices] = new_times
        self.buffer_ptr = (start_idx + n_new) % self.max_plot_points
        self.total_samples_received += n_new

        if self.is_fixed_run:
            # Show only collected samples so far (efficient downsample)
            available = self.total_samples_received
            stride = max(1, available // self.MAX_POINTS_TO_PLOT)
            idx_offsets = np.arange(0, available, stride, dtype=np.int64)
            indices = idx_offsets % self.max_plot_points
            x_data = self.time_buffer[indices]
            y_data = self.voltage_buffer[indices]
            self.plot_curve.setData(x=x_data, y=y_data)
        else:
            x_data, y_data = self._get_decimated_window()
            self.plot_widget.setXRange(0, self.time_scale_s, padding=0)
            self.plot_curve.setData(x=x_data, y=y_data)

        if new_data.size > 0:
            ymin = float(np.min(new_data))
            ymax = float(np.max(new_data))
            rms  = float(np.sqrt(np.mean(np.square(new_data))))
            p2p  = ymax - ymin
            self.stats_label.setText(
                f"min: {ymin:.3f} V\nmax: {ymax:.3f} V\nRMS: {rms:.3f} V\nP2P: {p2p:.3f} V")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.run_worker:
            self.run_worker.stop()
            self.run_worker.wait()
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker.wait()
        event.accept()

    def update_max_duration(self):
        desired_rate_k = self.rate_spinbox.value()
        desired_rate = desired_rate_k * 1000
        max_duration = MAX_VISIBLE_SAMPLES / desired_rate if desired_rate > 0 else 0
        self.effective_rate_label.setText(f"Max T: {max_duration:.2f} s")

    @QtCore.pyqtSlot(np.ndarray)
    def update_live_plot(self, new_data):
        """Append new data to the circular buffer and update the live plot efficiently."""
        if self.freeze_button.isChecked():
            return
        n_new = new_data.size
        if n_new == 0:
            return
        # Update buffers (same as before)
        self.sample_clock += n_new
        new_times = (self.sample_clock / self.sample_rate) - (np.arange(n_new, 0, -1) / self.sample_rate)
        start_idx = self.buffer_ptr
        indices = np.arange(start_idx, start_idx + n_new) % self.max_plot_points
        self.voltage_buffer[indices] = new_data
        self.time_buffer[indices] = new_times
        self.buffer_ptr = (start_idx + n_new) % self.max_plot_points
        self.total_samples_received += n_new

        # Get data for display
        if self.trigger_enabled:
            # Use triggered display
            x_data, y_data = self._get_triggered_window()
            
            # Set optimal X range based on detected period
            if self.trigger_enabled and len(x_data) > 0:
                # Use the actual data range instead of theoretical period calculation
                self.plot_widget.setXRange(0, x_data[-1], padding=0.02)
            elif len(x_data) > 0:
                self.plot_widget.setXRange(0, x_data[-1], padding=0.05)
        else:
            # Use normal rolling display
            x_data, y_data = self._get_decimated_window()
            self.plot_widget.setXRange(0, self.time_scale_s, padding=0)
            
        self.plot_curve.setData(x=x_data, y=y_data)

        # Update trigger visualization
        if self.trigger_enabled and len(y_data) > 0:
            self._update_trigger_display(x_data, y_data)
        else:
            # Clear trigger markers when not enabled
            self.trigger_markers.setData([], [])

        if y_data.size > 0:
            ymin = float(np.min(y_data))
            ymax = float(np.max(y_data))
            rms  = float(np.sqrt(np.mean(np.square(y_data))))
            p2p  = ymax - ymin
            self.stats_label.setText(
                f"min: {ymin:.3f} V\nmax: {ymax:.3f} V\nRMS: {rms:.3f} V\nP2P: {p2p:.3f} V")
                
    def _get_triggered_window(self):
        """Get triggered window of data for display"""
        # Get recent data for trigger analysis
        window_samples = int(self.time_scale_s * self.sample_rate * self.trigger_system.periods_to_display)
        available = min(window_samples, self.total_samples_received)
        
        if available < 100:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
            
        end_ptr = self.buffer_ptr
        start_ptr = (end_ptr - available + self.max_plot_points) % self.max_plot_points
        
        # Extract data for trigger analysis
        if start_ptr < end_ptr:
            voltage_data = self.voltage_buffer[start_ptr:end_ptr]
            time_data = self.time_buffer[start_ptr:end_ptr]
        else:
            voltage_data = np.concatenate((self.voltage_buffer[start_ptr:], self.voltage_buffer[:end_ptr]))
            time_data = np.concatenate((self.time_buffer[start_ptr:], self.time_buffer[:end_ptr]))
            
        # Get triggered window
        triggered_voltage, offset = self.trigger_system.get_triggered_window(voltage_data, window_periods=self.trigger_system.periods_to_display)
        
        if len(triggered_voltage) == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
            
        triggered_time = time_data[offset:offset+len(triggered_voltage)]
        
        # Center the display around the trigger point and scale appropriately
        if self.trigger_system.detected_period is not None:
            # Calculate optimal display window based on detected period
            period_time = self.trigger_system.detected_period / self.sample_rate
            display_periods = self.trigger_system.periods_to_display
            optimal_window_time = display_periods * period_time
            
            # Find trigger points in the triggered data
            triggers = self.trigger_system.find_triggers(triggered_voltage)
            
            if len(triggers) > 0:
                # Use the first trigger as the starting point for clean period display
                start_trigger_idx = triggers[0]
                start_time = triggered_time[start_trigger_idx] if start_trigger_idx < len(triggered_time) else triggered_time[0]
                
                # Calculate window bounds starting from the first trigger
                end_time = start_time + optimal_window_time
                
                # Find indices corresponding to this time window
                time_mask = (triggered_time >= start_time) & (triggered_time <= end_time)
                
                if np.any(time_mask):
                    final_voltage = triggered_voltage[time_mask]
                    final_time = triggered_time[time_mask]
                    
                    # Normalize time to start from 0
                    if len(final_time) > 0:
                        final_time = final_time - final_time[0]
                else:
                    # Fallback if masking fails
                    final_voltage = triggered_voltage
                    final_time = triggered_time - triggered_time[0]
            else:
                # No triggers found, use original data
                final_voltage = triggered_voltage
                final_time = triggered_time - triggered_time[0]
        else:
            # No period detected, use original approach
            final_voltage = triggered_voltage
            final_time = triggered_time - triggered_time[0] if len(triggered_time) > 0 else triggered_time
            
        # Downsample if needed
        if len(final_voltage) > self.MAX_POINTS_TO_PLOT:
            stride = len(final_voltage) // self.MAX_POINTS_TO_PLOT
            final_voltage = final_voltage[::stride]
            final_time = final_time[::stride]
            
        return final_time, final_voltage
        
    def _update_trigger_display(self, x_data, y_data):
        """Update trigger visualization on the plot"""
        if len(y_data) == 0:
            return
            
        # Update trigger level line
        self.trigger_line.setPos(self.trigger_system.level)
        
        # Find and display trigger points
        triggers = self.trigger_system.find_triggers(y_data)
        
        if len(triggers) > 0 and len(x_data) > max(triggers):
            trigger_times = x_data[triggers]
            trigger_voltages = y_data[triggers]
            self.trigger_markers.setData(trigger_times, trigger_voltages)
            
            # Update status with period info
            if self.trigger_system.detected_period is not None:
                period_samples = self.trigger_system.detected_period
                period_time = period_samples / self.sample_rate
                frequency = 1.0 / period_time if period_time > 0 else 0
                
                # Debug logging for frequency calculation
                if DEBUG_ENABLED:
                    logger.debug(f"Frequency calc: period_samples={period_samples}, sample_rate={self.sample_rate}, period_time={period_time}, freq={frequency}")
                
                # Format frequency appropriately
                if frequency >= 1000:
                    freq_str = f"{frequency/1000:.2f}kHz"
                elif frequency >= 1:
                    freq_str = f"{frequency:.1f}Hz"
                else:
                    freq_str = f"{frequency*1000:.1f}mHz"
                
                status_text = f"Status: {self.trigger_system.mode.title()} | Period: {period_time*1000:.1f}ms | Freq: {freq_str}"
                self.trigger_status_label.setText(status_text)
                
                # Update the trigger status color to indicate active triggering
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #90EE90; padding: 5px; }")
            else:
                self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | Searching...")
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")
        else:
            self.trigger_markers.setData([], [])
            if self.trigger_enabled:
                self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | No triggers")
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFB6C1; padding: 5px; }")

def main():
    """Main function to run the application."""
    logger.info("Attempting to connect to instrument...")
    try:
        client = connect(DEFAULT_HOST, 'laser-control')
        driver = CurrentRamp(client)
        logger.info("✅ Connected to %s at %s", INSTRUMENT_NAME, DEFAULT_HOST)
    except Exception as e:
        logger.error("❌ Connection failed: %s", e)
        logger.error("Please ensure the instrument is running and accessible.")
        return 1

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    main_win = MainWindow(driver)
    main_win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()