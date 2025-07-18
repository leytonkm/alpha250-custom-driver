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
DEFAULT_HOST = os.environ.get('HOST', '192.168.1.115')
INSTRUMENT_NAME = 'alpha15-laser-control'

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
    FS_ADC = 15_000_000  # 15 MHz for Alpha15 board (CORRECTED from earlier incorrect 240 MHz)
    target_rate = FS_ADC * run_seconds / MAX_VISIBLE_SAMPLES
    dec_rate = math.ceil(target_rate)
    if dec_rate < 10:
        dec_rate = 10  # Minimum decimation rate from config
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

    # New: switch ADC input range (0 = 2 Vpp, 1 = 8 Vpp)
    @command('CurrentRamp')
    def set_adc_input_range(self, range_sel):
        pass

    @command('CurrentRamp')
    def get_adc_input_range(self):
        return self.client.recv_uint32()

    @command('CurrentRamp')
    def select_adc_channel(self, channel):
        pass

# --- Triggering System ---

class TriggerSystem:
    """Simple, reliable oscilloscope-like triggering system."""
    
    def __init__(self):
        self.mode = 'auto'  # 'auto', 'peak', 'zero_cross', 'edge', 'hysteresis', 'photodiode'
        self.level = 0.0
        self.edge = 'rising'  # 'rising', 'falling'
        self.auto_level = True
        self.periods_to_display = 2.0
        
        # Simple period detection
        self.detected_period = None
        self.period_history = []
        self.max_period_history = 5  # Reduced for faster response
        
        # Template matching for consistent trigger selection
        self.trigger_template = None
        self.template_window = 50  # samples around trigger point
        self.template_update_counter = 0
        
        # Timeout tracking for automatic reset
        self.last_successful_trigger_time = None
        self.trigger_timeout_seconds = 3.0  # Reset if no triggers for 3 seconds
        
        # Signal preprocessing options for noisy photodiode signals
        self.enable_filtering = False  # Enable simple moving average filtering
        self.filter_window = 5  # Simple moving average window size
        
    def reset_trigger_system(self):
        """Reset trigger system - clear period history and detected period"""
        self.detected_period = None
        self.period_history = []
        print("Trigger system reset - period detection cleared")
        
    def set_trigger_mode(self, mode):
        """Set trigger mode and reset period detection"""
        self.mode = mode
        # Reset period detection when mode changes to avoid stale data
        self.reset_trigger_system()
        
    def set_trigger_level(self, level):
        """Set trigger level"""
        self.level = level
        self.auto_level = False
        
    def update_auto_level(self, data):
        """Update auto level only when explicitly called"""
        if self.auto_level and len(data) > 0:
            data_min, data_max = np.min(data), np.max(data)
            self.level = (data_min + data_max) / 2
            
    def find_peaks_simple(self, data):
        """Simple peak detection with error handling"""
        try:
            if len(data) < 10:
                return np.array([])
                
            data_range = np.max(data) - np.min(data)
            if data_range == 0:
                return np.array([])
                
            min_height = np.min(data) + 0.3 * data_range
            
            # Adaptive minimum distance based on detected period
            if self.detected_period is not None:
                # Use 1/4 of the detected period as minimum distance for all frequencies
                min_distance = max(5, self.detected_period // 4)  # Reduced minimum for high freq
                # Cap the minimum distance to prevent excessive spacing
                min_distance = min(min_distance, len(data) // 10)
            else:
                # Fallback: scale with data length
                min_distance = max(5, len(data) // 50)  # Reduced minimum for high freq
            
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(data, height=min_height, distance=min_distance)
            return peaks
            
        except Exception as e:
            print(f"Error in peak detection: {e}")
            return np.array([])
        
    def find_zero_crossings_simple(self, data):
        """Simple zero crossing detection with error handling"""
        try:
            if len(data) < 2:
                return np.array([])
                
            threshold = self.level
            
            if self.edge == 'rising':
                crossings = np.where((data[:-1] <= threshold) & (data[1:] > threshold))[0]
            elif self.edge == 'falling':
                crossings = np.where((data[:-1] >= threshold) & (data[1:] < threshold))[0]
            else:
                return np.array([])
                
            return crossings + 1
            
        except Exception as e:
            print(f"Error in zero crossing detection: {e}")
            return np.array([])
        
    def find_edges_simple(self, data):
        """Simple edge detection using derivative with error handling"""
        try:
            if len(data) < 3:
                return np.array([])
                
            derivative = np.diff(data)
            deriv_std = np.std(derivative)
            
            if deriv_std == 0:
                return np.array([])
                
            threshold = deriv_std  # Simple threshold
            
            # Adaptive minimum distance based on detected period
            if self.detected_period is not None:
                # Use 1/4 of the detected period as minimum distance for all frequencies
                min_distance = max(5, self.detected_period // 4)  # Reduced minimum for high freq
                # Cap the minimum distance to prevent excessive spacing
                min_distance = min(min_distance, len(data) // 10)
            else:
                # Fallback: scale with data length
                min_distance = max(5, len(data) // 50)  # Reduced minimum for high freq
            
            from scipy.signal import find_peaks
            if self.edge == 'rising':
                edges, _ = find_peaks(derivative, height=threshold, distance=min_distance)
            else:
                edges, _ = find_peaks(-derivative, height=threshold, distance=min_distance)
                
            return edges
            
        except Exception as e:
            print(f"Error in edge detection: {e}")
            return np.array([])
        
    def hysteresis_simple(self, data):
        """Simple hysteresis trigger"""
        if len(data) < 2:
            return np.array([])
            
        data_range = np.max(data) - np.min(data)
        if data_range == 0:
            return np.array([])
            
        # Fixed hysteresis amount - no adaptive complexity
        hyst_amount = 0.02 * data_range  # 2% of signal range
        low_thresh = self.level - hyst_amount
        high_thresh = self.level + hyst_amount
        
        state = 'low'
        triggers = []
        
        for i, sample in enumerate(data):
            if state == 'low' and sample > high_thresh:
                triggers.append(i)
                state = 'high'
            elif state == 'high' and sample < low_thresh:
                state = 'low'
                
        return np.array(triggers)
        
    def find_hysteresis_triggers(self, data):
        """Hysteresis triggering - enhanced for photodiode signals with error handling"""
        try:
            if len(data) < 10:
                return np.array([])
                
            # Enhanced hysteresis for photodiode signals
            data_range = np.max(data) - np.min(data)
            if data_range == 0:
                return np.array([])
                
            # More aggressive hysteresis for noisy photodiode signals
            # Use 10% instead of 5% for better noise immunity
            hysteresis = 0.10 * data_range
            high_thresh = self.level + hysteresis
            low_thresh = self.level - hysteresis
            
            # Edge-aware hysteresis with optional derivative gating
            triggers = []

            # Determine initial state based on actual signal level to avoid
            # frame-to-frame alternation when the first sample starts in the
            # opposite state.
            state = 'high' if data[0] > high_thresh else 'low'

            # Pre-compute derivative statistics for magnitude gating
            derivative_all = np.diff(data)
            deriv_threshold = np.std(derivative_all) * 0.5  # 0.5 σ – adapt to noise level

            for i in range(1, len(data)):
                deriv = derivative_all[i - 1]

                if self.edge == 'rising':
                    # Trigger on low→high crossing with sufficiently steep positive slope
                    if state == 'low' and data[i] > high_thresh and deriv > deriv_threshold:
                        triggers.append(i)
                        state = 'high'
                    elif state == 'high' and data[i] < low_thresh:
                        state = 'low'

                else:  # falling edge
                    # Trigger on high→low crossing with sufficiently steep negative slope
                    if state == 'high' and data[i] < low_thresh and deriv < -deriv_threshold:
                        triggers.append(i)
                        state = 'low'
                    elif state == 'low' and data[i] > high_thresh:
                        state = 'high'
            
            return np.array(triggers)
            
        except Exception as e:
            print(f"Error in hysteresis detection: {e}")
            return np.array([])

    def find_photodiode_triggers(self, data):
        """Enhanced trigger detection specifically designed for photodiode signals"""
        try:
            if len(data) < 20:
                return np.array([])
            
            # Step 1: Baseline tracking and removal
            baseline_window = min(100, len(data) // 10)
            if baseline_window > 0:
                # Use rolling minimum as baseline estimate (photodiodes usually have sharp rises from baseline)
                baseline = np.convolve(data, np.ones(baseline_window)/baseline_window, mode='same')
                # Shift baseline down slightly to account for noise
                baseline = baseline - 0.1 * np.std(data)
                corrected_data = data - baseline
            else:
                corrected_data = data - np.mean(data)
            
            # Step 2: Simple noise filtering (moving average)
            filter_window = max(3, min(10, len(data) // 100))
            if filter_window > 1:
                # Simple moving average filter
                filtered_data = np.convolve(corrected_data, np.ones(filter_window)/filter_window, mode='same')
            else:
                filtered_data = corrected_data
            
            # Step 3: Enhanced peak detection for photodiode pulses
            data_range = np.max(filtered_data) - np.min(filtered_data)
            if data_range == 0:
                return np.array([])
            
            # For photodiodes, look for sharp rises above baseline
            # Use derivative to find sharp transitions
            if len(filtered_data) > 2:
                derivative = np.diff(filtered_data)
                deriv_threshold = np.std(derivative) * 2  # 2 sigma threshold
                
                # Find points where derivative is high (sharp rise)
                rise_candidates = np.where(derivative > deriv_threshold)[0] + 1
                
                if len(rise_candidates) > 0:
                    # Filter by amplitude - must be significantly above baseline
                    amplitude_threshold = np.min(filtered_data) + 0.3 * data_range
                    valid_triggers = []
                    
                    for candidate in rise_candidates:
                        if candidate < len(filtered_data) and filtered_data[candidate] > amplitude_threshold:
                            valid_triggers.append(candidate)
                    
                    return np.array(valid_triggers)
            
            # Fallback to enhanced hysteresis if derivative method fails
            return self.find_hysteresis_triggers(data)
            
        except Exception as e:
            print(f"Error in photodiode trigger detection: {e}")
            return self.find_hysteresis_triggers(data)  # Fallback to hysteresis
        
    def update_auto_level(self, data):
        """Enhanced auto level for photodiode signals - tracks baseline better"""
        if self.auto_level and len(data) > 0:
            # For photodiode signals, use a more sophisticated auto-level
            # that tracks the baseline rather than just the midpoint
            
            # Use percentiles to be robust against outliers
            data_min = np.percentile(data, 5)   # 5th percentile as baseline
            data_max = np.percentile(data, 95)  # 95th percentile as peak
            
            # Set trigger level at 30% above baseline instead of midpoint
            # This works better for asymmetric photodiode pulses
            self.level = data_min + 0.3 * (data_max - data_min)
        
    def detect_period_simple(self, data):
        """Simple period detection using autocorrelation with error handling"""
        try:
            if len(data) < 50:  # Reduced from 100 for better low-frequency support
                return None
                
            # Check for data discontinuities that might indicate DMA restart
            # Skip period detection if we detect large jumps that suggest buffer issues
            if len(data) > 100:
                data_diff = np.diff(data)
                max_jump = np.max(np.abs(data_diff))
                data_range = np.max(data) - np.min(data)
                if data_range > 0 and max_jump > 0.5 * data_range:
                    # Large jump detected - likely buffer discontinuity, skip period detection
                    return None
                
            # Timeout protection
            import time
            start_time = time.time()
            max_processing_time = 0.05  # 50ms max for autocorrelation
            
            # Limit data size to prevent excessive computation
            if len(data) > 200000:  # > 2 seconds at 100kHz
                print(f"Warning: Data too large for autocorrelation ({len(data)} samples), skipping")
                return None
                
            # For very long data (likely low frequency), use decimation for efficiency
            # Adjust threshold for 3-5Hz range
            if len(data) > 20000:  # > 0.2 seconds at 100kHz (covers 3-5Hz range better)
                # Decimate by factor of 5 for better resolution in 3-5Hz range
                decimated_data = data[::5]
                decimation_factor = 5
            elif len(data) > 50000:  # > 0.5 seconds at 100kHz
                # Decimate by factor of 10 for very low frequencies
                decimated_data = data[::10]
                decimation_factor = 10
            else:
                decimated_data = data
                decimation_factor = 1
                
            # Check timeout after decimation
            if time.time() - start_time > max_processing_time:
                print("Warning: Autocorrelation timeout during decimation")
                return None
                
            # Simple autocorrelation
            data_centered = decimated_data - np.mean(decimated_data)
            
            # Prevent division by zero
            if np.std(data_centered) == 0:
                return None
                
            autocorr = np.correlate(data_centered, data_centered, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            if autocorr[0] == 0:
                return None
                
            autocorr = autocorr / autocorr[0]
            
            # Check timeout after autocorrelation
            if time.time() - start_time > max_processing_time:
                print("Warning: Autocorrelation timeout during correlation")
                return None
            
            # Adaptive search range based on data length
            min_period = 3  # Reduced minimum period for better 3-5Hz detection
            max_period = min(len(decimated_data) // 2, len(autocorr) - 1)  # Increased to 1/2 of data length
            
            if max_period <= min_period:
                return None
                
            search_range = autocorr[min_period:max_period]
            if len(search_range) < 5:
                return None
                
            from scipy.signal import find_peaks
            # Lower threshold and smaller distance for better detection in 3-5Hz range
            peaks, _ = find_peaks(search_range, height=0.15, distance=3)
            
            if len(peaks) > 0:
                # Convert back to original sample rate
                detected_period = (peaks[0] + min_period) * decimation_factor
                return detected_period
                
            return None
            
        except Exception as e:
            print(f"Error in period detection: {e}")
            return None
        
    def update_period_estimate(self, new_period):
        """Update period estimate with simple smoothing and validation"""
        if new_period is not None and new_period > 0:
            # Validate new period against existing history
            if len(self.period_history) > 0:
                median_period = np.median(self.period_history)
                # More conservative period change detection to prevent drift
                period_change_ratio = abs(new_period - median_period) / median_period
                if period_change_ratio > 0.3:  # Tightened from 0.5 to 0.3
                    # Only reset if we see multiple large changes in a row
                    if not hasattr(self, 'period_change_count'):
                        self.period_change_count = 0
                    self.period_change_count += 1
                    
                    if self.period_change_count >= 3:  # Require 3 consecutive large changes
                        print(f"Consistent period change detected: {median_period:.0f} -> {new_period:.0f} ({period_change_ratio:.1%}), resetting history")
                        self.period_history = []
                        self.period_change_count = 0
                        self.trigger_template = None  # Reset template too
                    else:
                        print(f"Period change {self.period_change_count}/3: {median_period:.0f} -> {new_period:.0f} ({period_change_ratio:.1%})")
                        return  # Don't add this period to history yet
                else:
                    self.period_change_count = 0  # Reset counter on stable periods
             
            self.period_history.append(new_period)
            if len(self.period_history) > self.max_period_history:
                self.period_history.pop(0)
                
            # Simple median for robustness
            self.detected_period = int(np.median(self.period_history))
            
    def find_triggers(self, data):
        """Main trigger detection - simplified and predictable with error handling"""
        try:
            if len(data) < 50:
                return np.array([])
                
            # Timeout protection - limit processing time
            import time
            start_time = time.time()
            max_processing_time = 0.1  # 100ms max processing time
            
            # Apply signal preprocessing for noisy photodiode signals
            processed_data = self.preprocess_signal(data)
            
            # Get triggers based on mode
            if self.mode == 'auto':
                triggers = self._auto_trigger_simple(processed_data)
            elif self.mode == 'peak':
                triggers = self.find_peaks_simple(processed_data)
            elif self.mode == 'zero_cross':
                triggers = self.find_zero_crossings_simple(processed_data)
            elif self.mode == 'edge':
                triggers = self.find_edges_simple(processed_data)
            elif self.mode == 'hysteresis':
                triggers = self.find_hysteresis_triggers(processed_data)
            elif self.mode == 'photodiode':
                triggers = self.find_photodiode_triggers(processed_data)
            else:
                triggers = np.array([])
            
            # Check for timeout
            if time.time() - start_time > max_processing_time:
                print(f"Warning: Trigger processing timeout for mode {self.mode}")
                return np.array([])
                
            # Ensure triggers is a numpy array
            if not isinstance(triggers, np.ndarray):
                triggers = np.array(triggers)
                
            # Sanity check on triggers
            if len(triggers) > len(data) // 2:
                print(f"Warning: Too many triggers detected ({len(triggers)}), clearing")
                triggers = np.array([])
                
            # Adaptive holdoff based on detected period
            if len(triggers) > 1:
                # Period-based hold-off: default to 80 % of detected period.
                # This locks the window to the first qualifying edge each cycle
                # and prevents the view from jumping to secondary peaks.
                if self.detected_period is not None:
                    # Use 95 % of detected period so we treat the whole cycle as one cluster
                    min_distance = int(0.95 * self.detected_period)
                    # Ensure a sensible lower bound so very high-freq signals still work
                    min_distance = max(min_distance, 5)
                else:
                    # Fallback when period not known yet – pick a conservative constant
                    min_distance = 50

                # Peak-Anchoring: Stably select the trigger leading to the highest peak in each cycle.
                filtered_triggers = []
                i = 0
                while i < len(triggers):
                    # 1. Group candidate triggers into clusters based on the detected period
                    cluster_start_idx = i
                    cluster_end_idx = i
                    while (cluster_end_idx + 1 < len(triggers) and
                           triggers[cluster_end_idx + 1] - triggers[cluster_start_idx] < min_distance):
                        cluster_end_idx += 1
                    
                    current_cluster = triggers[cluster_start_idx : cluster_end_idx + 1]

                    if len(current_cluster) > 0:
                        # 2. Find the highest peak associated with this trigger cluster
                        start_search = current_cluster[0]
                        end_search = min(start_search + min_distance, len(processed_data))
                        
                        # Find index of max value within this window slice, offsetting back to full data coordinates
                        local_max_idx = start_search + np.argmax(processed_data[start_search:end_search])

                        # 3. Select the trigger from the cluster that is closest to (and before) this peak
                        best_trigger = current_cluster[0]
                        min_dist_to_peak = np.inf
                        for t in current_cluster:
                            # Ensure trigger is on the leading edge of the peak
                            if t <= local_max_idx:
                                dist = local_max_idx - t
                                if dist < min_dist_to_peak:
                                    min_dist_to_peak = dist
                                    best_trigger = t

                        filtered_triggers.append(best_trigger)

                    # 4. Move to the next cluster
                    i = cluster_end_idx + 1

                triggers = np.array(filtered_triggers)
                
            # Update period estimate
            if len(triggers) >= 2:
                periods = np.diff(triggers)
                if len(periods) > 0:
                    avg_period = np.median(periods)
                    self.update_period_estimate(avg_period)
                    # Update successful trigger time
                    import time
                    self.last_successful_trigger_time = time.time()
            elif len(triggers) == 0:
                # Check for timeout and reset if needed
                import time
                current_time = time.time()
                if (self.last_successful_trigger_time is not None and 
                    current_time - self.last_successful_trigger_time > self.trigger_timeout_seconds):
                    print("Trigger timeout - automatically resetting trigger system")
                    self.reset_trigger_system()
                    self.last_successful_trigger_time = current_time
                
                # Only try autocorrelation if no triggers found and data is reasonable size
                if len(data) < 100000:  # Prevent excessive computation
                    period = self.detect_period_simple(data)
                    self.update_period_estimate(period)
            else:
                # Single trigger found - update time but don't update period
                import time
                self.last_successful_trigger_time = time.time()
                    
            return triggers
            
        except Exception as e:
            print(f"Error in trigger detection: {e}")
            return np.array([])  # Return empty array on any error
        
    def _auto_trigger_simple(self, data):
        """Enhanced auto trigger - tries multiple methods including photodiode-specific detection"""
        # For low frequencies (detected period > 0.2 seconds), try hysteresis first
        if self.detected_period is not None and self.detected_period > 20000:  # > 0.2 seconds at 100kHz (< 5Hz)
            # Try hysteresis first for low frequencies (including 3-5Hz range)
            hysteresis = self.find_hysteresis_triggers(data)
            if len(hysteresis) >= 1:
                return hysteresis
        
        # Try photodiode-specific detection first for irregular/noisy signals
        photodiode_triggers = self.find_photodiode_triggers(data)
        if len(photodiode_triggers) >= 2:
            return photodiode_triggers
        
        # Try peaks detection
        peaks = self.find_peaks_simple(data)
        if len(peaks) >= 2:
            return peaks
            
        # Try zero crossings
        crossings = self.find_zero_crossings_simple(data)
        if len(crossings) >= 2:
            return crossings
            
        # Try enhanced hysteresis as fallback
        hysteresis = self.find_hysteresis_triggers(data)
        if len(hysteresis) >= 1:
            return hysteresis
            
        return np.array([])
        
    def get_triggered_window(self, data, triggers=None, window_periods=None):
        """Simple triggered window that actually uses triggers"""
        if window_periods is None:
            window_periods = self.periods_to_display
            
        # If no triggers provided, find them
        if triggers is None:
            return data[len(data)//4:3*len(data)//4], len(data)//4  # Fallback to center
            
        if len(triggers) == 0:
            return data[len(data)//4:3*len(data)//4], len(data)//4  # Fallback to center
            
        # Use the first trigger as the reference point
        trigger_idx = triggers[0]
        
        # Determine window size
        if self.detected_period is not None:
            window_size = int(window_periods * self.detected_period)
            # Ensure window size is reasonable
            window_size = min(window_size, len(data) // 2)
            window_size = max(window_size, 100)  # Minimum window size
        else:
            window_size = len(data) // 3  # Default window
            
        # Start window AT the trigger point for consistent alignment
        start = trigger_idx
        end = min(len(data), start + window_size)
        
        # If we don't have enough data after the trigger, try the previous trigger
        if end - start < window_size and len(triggers) > 1:
            # Try the previous trigger if available
            prev_trigger = triggers[0] if triggers[0] > window_size else triggers[1] if len(triggers) > 1 else triggers[0]
            start = max(0, prev_trigger)
            end = min(len(data), start + window_size)
        
        # Ensure we have a minimum window
        if end - start < 100:
            start = max(0, trigger_idx - 50)
            end = min(len(data), start + 100)
        
        return data[start:end], start
        
    def get_status_info(self):
        """Get trigger status information"""
        return {
            'mode': self.mode,
            'level': self.level,
            'edge': self.edge,
            'period': self.detected_period,
            'auto_level': self.auto_level,
            'periods_to_display': self.periods_to_display
        }

    def preprocess_signal(self, data):
        """Apply signal preprocessing for noisy photodiode signals"""
        if not self.enable_filtering or len(data) < self.filter_window:
            return data
            
        try:
            # Simple moving average filter for noise reduction
            filtered = np.convolve(data, np.ones(self.filter_window)/self.filter_window, mode='same')
            return filtered.astype(np.float32)
        except Exception as e:
            print(f"Error in signal preprocessing: {e}")
            return data

    def _select_best_trigger_with_template(self, candidate_triggers, data):
        """Select trigger using template matching for maximum consistency"""
        if len(candidate_triggers) == 1:
            return candidate_triggers[0]
            
        # If no template yet, pick the steepest and create template
        if self.trigger_template is None:
            data_deriv = np.diff(data)
            best_idx = candidate_triggers[0]
            best_slope = -np.inf
            
            for t in candidate_triggers:
                raw_slope = data_deriv[t - 1] if 0 < t < len(data_deriv) else data_deriv[-1]
                slope = raw_slope if self.edge == 'rising' else -raw_slope
                if slope > best_slope:
                    best_slope = slope
                    best_idx = t
            
            # Create template around this trigger
            self._update_template(best_idx, data)
            return best_idx
        
        # Use template matching to find most similar trigger
        best_idx = candidate_triggers[0]
        best_correlation = -1
        
        for t in candidate_triggers:
            correlation = self._calculate_template_correlation(t, data)
            if correlation > best_correlation:
                best_correlation = correlation
                best_idx = t
        
        # Occasionally update template with best match
        self.template_update_counter += 1
        if self.template_update_counter >= 10:  # Update every 10th trigger
            self._update_template(best_idx, data)
            self.template_update_counter = 0
        
        return best_idx
    
    def _update_template(self, trigger_idx, data):
        """Update trigger template around the given trigger point"""
        try:
            start = max(0, trigger_idx - self.template_window // 2)
            end = min(len(data), trigger_idx + self.template_window // 2)
            
            if end - start >= self.template_window // 2:  # Ensure minimum template size
                template = data[start:end].copy()
                # Normalize template
                template = template - np.mean(template)
                if np.std(template) > 0:
                    template = template / np.std(template)
                    self.trigger_template = template
        except Exception as e:
            print(f"Error updating template: {e}")
    
    def _calculate_template_correlation(self, trigger_idx, data):
        """Calculate correlation between template and data around trigger point"""
        try:
            if self.trigger_template is None:
                return 0
                
            start = max(0, trigger_idx - self.template_window // 2)
            end = min(len(data), trigger_idx + self.template_window // 2)
            
            if end - start < len(self.trigger_template):
                return 0
                
            window = data[start:start + len(self.trigger_template)]
            # Normalize window
            window = window - np.mean(window)
            if np.std(window) > 0:
                window = window / np.std(window)
                # Calculate correlation
                correlation = np.corrcoef(self.trigger_template, window)[0, 1]
                return correlation if not np.isnan(correlation) else 0
            return 0
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0

# --- Live Window Worker (infinite rolling oscilloscope) ---

class LiveWorker(QtCore.QThread):
    """Continuously fetch the last `window_seconds` of data each refresh."""
    window_ready = QtCore.pyqtSignal(np.ndarray)
    status_changed = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    dma_error_occurred = QtCore.pyqtSignal()  # New signal for DMA errors

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
                            self.dma_error_occurred.emit()  # Notify GUI to reset trigger system
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

        # --- Initial ADC Setup ---
        # Select ADC channel 0 (RFADC0) and set range
        self.driver.select_adc_channel(0)
        self.driver.set_adc_input_range(1)  # 8 Vpp range (for better signal visibility)
        
        # Initialize current ADC range to match the driver setting
        self.current_adc_range = 1  # 8 Vpp
        
        # Check actual ADC range from driver and sync UI
        try:
            actual_range = self.driver.get_adc_input_range()
            self.current_adc_range = actual_range
            print(f"DEBUG: Driver ADC range is {actual_range} ({'2 Vpp' if actual_range == 0 else '8 Vpp'})")
        except Exception as e:
            print(f"DEBUG: Could not get ADC range from driver: {e}")
            # Force set to 8 Vpp
            self.driver.set_adc_input_range(1)
            self.current_adc_range = 1
            print("DEBUG: Forced ADC range to 8 Vpp")
        
        # --- Data Buffers ---
        self.time_scale_s = 5.0 # Default time window to display
        self.sample_rate = self.driver.get_decimated_sample_rate()
        
        # --- Trigger System ---
        self.trigger_system = TriggerSystem()
        self.trigger_enabled = False
        
        # Stored trigger data to avoid recursive calls
        self._last_triggers = np.array([])
        self._last_trigger_data = np.array([])
        self._last_time_data = np.array([])
        
        # Initialize trigger system timing
        import time
        self.trigger_system.last_successful_trigger_time = time.time()

        # Allocate live buffers sized to current time window plus safety margin
        # Set up live streaming buffers
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

        # Set initial time scale to match dropdown default
        self.set_time_scale(5.0)
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
        streaming_layout.addWidget(self.start_button)
        
        # Time Scale dropdown
        ts_layout = QtWidgets.QHBoxLayout()
        ts_layout.addWidget(QtWidgets.QLabel("Time Scale:"))
        self.time_scale_combo = QtWidgets.QComboBox()
        
        # Define time scale options (value in seconds, display text)
        self.time_scale_options = [
            (0.001, "1 ms"),
            (0.002, "2 ms"), 
            (0.005, "5 ms"),
            (0.01, "10 ms"),
            (0.02, "20 ms"),
            (0.05, "50 ms"),
            (0.1, "100 ms"),
            (0.2, "200 ms"),
            (0.5, "500 ms"),
            (1.0, "1 s"),
            (2.0, "2 s"),
            (5.0, "5 s"),
            (10.0, "10 s"),
            (20.0, "20 s"),
            (50.0, "50 s"),
        ]
        
        # Populate dropdown and set default
        for value, text in self.time_scale_options:
            self.time_scale_combo.addItem(text, value)
        
        # Set default to 5 seconds
        default_index = next(i for i, (val, _) in enumerate(self.time_scale_options) if val == 5.0)
        self.time_scale_combo.setCurrentIndex(default_index)
        
        ts_layout.addWidget(self.time_scale_combo)
        streaming_layout.addLayout(ts_layout)
        
        # Autoscale button
        self.autoscale_button = QtWidgets.QPushButton("Autoscale Voltage")
        self.autoscale_button.setToolTip("Automatically adjust plot Y-axis to fit the current signal range")
        self.autoscale_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:checked {
                background-color: #FFB6C1;
                border-color: #FF69B4;
            }
        """)
        streaming_layout.addWidget(self.autoscale_button)
        
        # ADC Range control - segmented control style
        adc_range_layout = QtWidgets.QHBoxLayout()
        adc_range_layout.addWidget(QtWidgets.QLabel("ADC Range:"))
        
        # Create button group for mutual exclusion
        self.range_button_group = QtWidgets.QButtonGroup()
        
        # Segmented control container
        range_container = QtWidgets.QWidget()
        range_button_layout = QtWidgets.QHBoxLayout(range_container)
        range_button_layout.setContentsMargins(0, 0, 0, 0)
        range_button_layout.setSpacing(0)
        
        # 2V button (left side of segmented control)
        self.range_2v_button = QtWidgets.QPushButton("2V")
        self.range_2v_button.setCheckable(True)
        self.range_2v_button.setFixedHeight(28)
        self.range_2v_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #aeaeae;
                border-right: 1px solid #4A90E2;
                border-top-left-radius: 6px;
                border-bottom-left-radius: 6px;
                border-top-right-radius: 0px;
                border-bottom-right-radius: 0px;
                                 color: #aeaeae;
=            }
            QPushButton:checked {
                background-color: #aeaeae;
                color: black;
            }
            QPushButton:hover:!checked {
                background-color: #E3F2FD;
            }
        """)
        
        # 8V button (right side of segmented control)
        self.range_8v_button = QtWidgets.QPushButton("8V")
        self.range_8v_button.setCheckable(True)
        self.range_8v_button.setFixedHeight(28)
        self.range_8v_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #aeaeae;
                border-left: 1px solid #aeaeae;
                border-top-left-radius: 0px;
                border-bottom-left-radius: 0px;
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                                 color: #aeaeae;
=            }
            QPushButton:checked {
                background-color: #aeaeae;
                color: black;
            }
            QPushButton:hover:!checked {
                background-color: #FFF3E0;
            }
        """)
        
        # Add buttons to group and layout with equal stretching
        self.range_button_group.addButton(self.range_2v_button, 0)
        self.range_button_group.addButton(self.range_8v_button, 1)
        range_button_layout.addWidget(self.range_2v_button, 1)  # stretch factor 1
        range_button_layout.addWidget(self.range_8v_button, 1)  # stretch factor 1
        
        adc_range_layout.addWidget(range_container)
        
        # Set initial state based on driver setting
        if self.current_adc_range == 0:
            self.range_2v_button.setChecked(True)
        else:
            self.range_8v_button.setChecked(True)
            
        streaming_layout.addLayout(adc_range_layout)
        
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
            }
            QPushButton:checked {
                background-color: #afd6fa;
                border-color: #4da9fe;
            }
        """)
        trigger_layout.addWidget(self.freeze_button)
        
        # Trigger mode selection
        trigger_mode_layout = QtWidgets.QHBoxLayout()
        trigger_mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        self.trigger_mode_combo = QtWidgets.QComboBox()
        self.trigger_mode_combo.addItems(['Auto', 'Peak', 'Zero Cross', 'Edge', 'Hysteresis', 'Photodiode'])
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
        
        # Auto level and reset buttons (compact, same line)
        trigger_buttons_layout = QtWidgets.QHBoxLayout()
        self.auto_level_button = QtWidgets.QPushButton("Auto Level")
        self.auto_level_button.setMinimumWidth(95)
        self.auto_level_button.setMaximumWidth(95)
        self.auto_level_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #aaa;
            }
        """)
        trigger_buttons_layout.addWidget(self.auto_level_button)
        
        self.reset_trigger_button = QtWidgets.QPushButton("Force Reset")
        self.reset_trigger_button.setToolTip("Force reset trigger system and clear all period detection")
        self.reset_trigger_button.setMinimumWidth(95)
        self.reset_trigger_button.setMaximumWidth(95)
        self.reset_trigger_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 3px;
                padding: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #aaa;
            }
        """)
        trigger_buttons_layout.addWidget(self.reset_trigger_button)
        
        trigger_layout.addLayout(trigger_buttons_layout)
        
        # Trigger status display
        self.trigger_status_label = QtWidgets.QLabel("Status: Disabled")
        self.trigger_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.trigger_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        trigger_layout.addWidget(self.trigger_status_label)
        
        # Signal preprocessing controls for noisy photodiode signals
        preprocessing_layout = QtWidgets.QHBoxLayout()
        self.enable_filtering_checkbox = QtWidgets.QCheckBox("Noise Filter")
        self.enable_filtering_checkbox.setToolTip("Enable simple moving average filter for noisy photodiode signals")
        preprocessing_layout.addWidget(self.enable_filtering_checkbox)
        
        self.filter_window_spinbox = QtWidgets.QSpinBox()
        self.filter_window_spinbox.setRange(3, 20)
        self.filter_window_spinbox.setValue(5)
        self.filter_window_spinbox.setToolTip("Filter window size (higher = more smoothing)")
        self.filter_window_spinbox.setEnabled(False)  # Initially disabled
        preprocessing_layout.addWidget(QtWidgets.QLabel("Size:"))
        preprocessing_layout.addWidget(self.filter_window_spinbox)
        
        trigger_layout.addLayout(preprocessing_layout)
        
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

        self.run_button = QtWidgets.QPushButton("Start Run")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #f0f0f0;
                border: 2px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:checked {
                background-color: #FFE4B5;
                border-color: #FFA500;
            }
        """)
        run_layout.addWidget(self.run_button, 1, 0, 1, 2)
        self.effective_rate_label = QtWidgets.QLabel("Rate: -- kHz | Max: -- s")
        self.effective_rate_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.effective_rate_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        run_layout.addWidget(self.effective_rate_label, 2, 0, 1, 2)
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setTextVisible(False)
        run_layout.addWidget(self.run_progress_bar, 3, 0, 1, 2)
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
        
        # Set initial Y range based on actual ADC range from driver
        # (self.current_adc_range is set during initialization)
        self.update_plot_y_range()
        
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

        main_layout.addLayout(plots_container)
        main_layout.addWidget(control_panel)

        # --- Status Bar ---
        self.status_bar = self.statusBar()



    def connect_signals(self):
        """Connect UI signals to slots."""
        self.start_button.toggled.connect(self.toggle_streaming)
        self.run_button.clicked.connect(self.start_fixed_run)
        self.time_scale_combo.currentIndexChanged.connect(self.on_time_scale_changed)

        # Update max duration label dynamically
        self.duration_spinbox.valueChanged.connect(self.update_max_duration)

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
        self.reset_trigger_button.clicked.connect(self.reset_trigger_system)
        self.range_button_group.buttonClicked.connect(self.on_adc_range_button_clicked)
        self.autoscale_button.clicked.connect(self.on_autoscale_voltage)

        # Connect signal preprocessing controls
        self.enable_filtering_checkbox.toggled.connect(self.on_filtering_enabled_changed)
        self.filter_window_spinbox.valueChanged.connect(self.on_filter_window_changed)

        # initialise max-duration label
        QtCore.QTimer.singleShot(0, self.update_max_duration)
        
    def on_time_scale_changed(self):
        """Handle time scale dropdown change"""
        current_index = self.time_scale_combo.currentIndex()
        if current_index >= 0:
            selected_value = self.time_scale_options[current_index][0]
            self.set_time_scale(selected_value)

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
            # Check if streaming is active before auto-starting
            if hasattr(self, 'live_worker') and self.live_worker is not None and self.live_worker.isRunning():
                # Streaming is already active
                self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | Searching...")
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")
            else:
                # Auto-start live view when trigger is enabled
                if not self.start_button.isChecked():
                    self.start_button.setChecked(True)
                    self.start_streaming()
                self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | Starting...")
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")
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
            'Hysteresis': 'hysteresis',
            'Photodiode': 'photodiode'
        }
        self.trigger_system.set_trigger_mode(mode_map[mode_text])
        
        if self.trigger_enabled:
            self.trigger_status_label.setText(f"Status: {mode_text} | Searching...")
            self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")
            
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
                # Update the trigger system's auto level
                self.trigger_system.update_auto_level(recent_data)
                # Update the GUI to reflect the new level
                self.trigger_level_spinbox.setValue(self.trigger_system.level)

    def on_periods_changed(self):
        """Update the number of periods to display in the triggered window."""
        self.trigger_system.periods_to_display = self.periods_spinbox.value()
        
    def on_autoscale_voltage(self):
        """Automatically scale the Y-axis to fit the current visible data."""
        try:
            x_data, y_data = self.plot_curve.getData()
            if x_data is not None and y_data is not None and len(y_data) > 0:
                y_min = float(np.min(y_data))
                y_max = float(np.max(y_data))
                
                # Add 5% margin on each side
                y_range = y_max - y_min
                if y_range > 0:
                    margin = y_range * 0.05
                    self.plot_widget.setYRange(y_min - margin, y_max + margin)
                    self.status_bar.showMessage(f"Autoscaled voltage: {y_min:.3f}V to {y_max:.3f}V")
                else:
                    # Handle flat signal case
                    margin = 0.1 if abs(y_min) < 0.001 else abs(y_min) * 0.1
                    self.plot_widget.setYRange(y_min - margin, y_min + margin)
                    self.status_bar.showMessage(f"Autoscaled flat signal: {y_min:.3f}V ±{margin:.3f}V")
            else:
                self.status_bar.showMessage("No data available for autoscaling")
        except Exception as e:
            self.status_bar.showMessage(f"Autoscale failed: {str(e)}")

    @QtCore.pyqtSlot(bool)
    def toggle_streaming(self, checked):
        """Toggle live streaming on/off."""
        if checked:
            # Stop any running fixed-duration acquisition first
            if self.run_worker and self.run_worker.isRunning():
                self.run_worker.stop()
                self.run_worker.wait()
                print("DEBUG: Stopped run worker before starting live stream")
            self.start_streaming()
        else:
            self.stop_streaming()

    def start_streaming(self):
        """Start live streaming with the LiveWorker thread."""
        if self.live_worker is not None and self.live_worker.isRunning():
            return  # Already running

        self.reset_and_clear_buffers()
        
        # Calculate guard points based on current time scale
        guard_pts = 8 * 2048  # 8 DMA descriptors as safety margin
        
        self.live_worker = LiveWorker(
            self.driver, 
            self.sample_rate, 
            self.time_scale_s, 
            guard_pts, 
            UPDATE_INTERVAL_MS
        )
        self.live_worker.window_ready.connect(self.update_live_plot)
        self.live_worker.status_changed.connect(self.status_bar.showMessage)
        self.live_worker.finished.connect(self.on_live_worker_finished)
        self.live_worker.dma_error_occurred.connect(self.on_dma_error_occurred)
        self.live_worker.start()
        
        # Update trigger status if trigger is enabled
        if self.trigger_enabled:
            self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | Searching...")
            self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")

    def stop_streaming(self):
        """Stop the live acquisition worker."""
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker.wait() # Wait for thread to finish cleanly
        
        # Ensure DMA is properly stopped on hardware side
        try:
            self.driver.stop_adc_streaming()
            print("DEBUG: DMA streaming stopped on hardware")
        except Exception as e:
            print(f"DEBUG: Error stopping DMA: {e}")

    def start_fixed_run(self):
        """Start the data acquisition worker for a fixed duration."""
        # First, ensure any live streaming is completely stopped
        if self.live_worker and self.live_worker.isRunning():
            self.stop_streaming()
            print("DEBUG: Stopped live streaming before starting run")
        
        # Ensure DMA is stopped before starting new acquisition
        try:
            self.driver.stop_adc_streaming()
            print("DEBUG: Ensured DMA is stopped before run")
        except Exception as e:
            print(f"DEBUG: Error ensuring DMA stop: {e}")
        
        # Use current sample rate (no user rate selection)
        # Check that requested duration fits in max points
        max_duration = MAX_VISIBLE_SAMPLES / self.sample_rate
        if self.run_duration > max_duration:
            QtWidgets.QMessageBox.warning(self, "Duration too long", 
                f"At {self.sample_rate/1000:.0f} kHz the maximum duration is {max_duration:.2f} s.")
            return
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
        self.time_scale_combo.setEnabled(False)
        
        self.plot_widget.setXRange(0, self.run_duration)
        self.run_progress_bar.setValue(0)
        
        self.reset_and_clear_buffers()
        
        samples_per_update = max(DEFAULT_SAMPLES_PER_UPDATE, int(0.1 * self.sample_rate))
        self.run_worker = RunWorker(self.driver, samples_per_update=samples_per_update, samples_to_collect=samples_to_collect)
        self.connect_run_worker_signals()
        self.run_worker.progress_updated.connect(self.run_progress_bar.setValue)
        self.run_worker.start()
        
        print(f"DEBUG: Started fixed run - {self.run_duration}s at {self.sample_rate/1e3:.1f} kHz")

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
        self.run_worker.data_ready.connect(self.update_main_plot_from_run)
        self.run_worker.status_changed.connect(self.status_bar.showMessage)
        self.run_worker.finished.connect(self.on_run_worker_finished)

    def on_run_worker_finished(self):
        """Called when a fixed duration run completes."""
        self.run_worker = None
        self.start_button.setEnabled(True)
        self.start_button.setChecked(False)
        self.run_button.setEnabled(True)
        self.duration_spinbox.setEnabled(True)
        self.time_scale_combo.setEnabled(True)

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
                # Ensure X and Y arrays are the same length
                visible_samples = min(visible_samples, len(self.voltage_buffer))
                y_data = self.voltage_buffer[:visible_samples]
                x_data = np.arange(visible_samples) / self.sample_rate
                self.plot_curve.setData(x=x_data, y=y_data)
                # update duration in case of slight mismatch
                run_duration_actual = visible_samples / self.sample_rate
                self.plot_widget.setXRange(0, run_duration_actual, padding=0.05)
            else:
                self.plot_curve.setData(x=[], y=[])

            self.status_bar.showMessage(f"Run of {self.run_duration}s finished. {visible_samples} samples collected.")
            self.run_progress_bar.setValue(100)

            # Data is displayed in the main plot (no separate run plot)
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
        # Format time scale display with appropriate units
        if scale_s >= 1.0:
            time_text = f"{scale_s:.0f}s" if scale_s == int(scale_s) else f"{scale_s:.1f}s"
        else:
            time_text = f"{scale_s*1000:.0f}ms"
        self.status_bar.showMessage(f"Time scale set to {time_text} (rate {self.sample_rate/1e3:.1f} kHz)")

    @QtCore.pyqtSlot(np.ndarray)
    def update_main_plot_from_run(self, new_data):
        """Update main plot during fixed-duration runs (no separate run plot)."""
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
        current_rate = self.sample_rate if hasattr(self, 'sample_rate') else 75000  # Default 75kHz
        max_duration = MAX_VISIBLE_SAMPLES / current_rate if current_rate > 0 else 0
        
        # Debug: Show both expected and actual rates
        if hasattr(self, 'driver'):
            try:
                actual_decimation = self.driver.get_decimation_rate()
                expected_rate = 15_000_000 / (2.0 * actual_decimation)
                self.effective_rate_label.setText(f"Rate: {current_rate/1000:.0f} kHz (dec={actual_decimation}) | Max: {max_duration:.2f} s")
            except:
                self.effective_rate_label.setText(f"Rate: {current_rate/1000:.0f} kHz | Max: {max_duration:.2f} s")
        else:
            self.effective_rate_label.setText(f"Rate: {current_rate/1000:.0f} kHz | Max: {max_duration:.2f} s")

    @QtCore.pyqtSlot(np.ndarray)
    def update_live_plot(self, new_data):
        """Append new data to the circular buffer and update the live plot efficiently."""
        if self.freeze_button.isChecked():
            return
        n_new = new_data.size
        if n_new == 0:
            return
        # Update buffers 
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
            
            # Set X range based on triggered window duration
            if len(x_data) > 0:
                # For triggered mode, show the exact window we calculated
                self.plot_widget.setXRange(0, x_data[-1], padding=0.02)
            else:
                # Fallback if no triggered data
                if self.trigger_system.detected_period is not None:
                    period_time = self.trigger_system.detected_period / self.sample_rate
                    total_time = self.trigger_system.periods_to_display * period_time
                    self.plot_widget.setXRange(0, total_time, padding=0.02)
                else:
                    self.plot_widget.setXRange(0, 1.0, padding=0.02)  # 1 second default
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
            
            # Add period and frequency info to live stats if trigger is enabled
            if self.trigger_enabled and self.trigger_system.detected_period is not None:
                period_samples = self.trigger_system.detected_period
                period_time = period_samples / self.sample_rate
                frequency = 1.0 / period_time if period_time > 0 else 0
                
                # Format frequency appropriately
                if frequency >= 1000:
                    freq_str = f"{frequency/1000:.2f}kHz"
                elif frequency >= 1:
                    freq_str = f"{frequency:.1f}Hz"
                else:
                    freq_str = f"{frequency*1000:.1f}mHz"
                
                self.stats_label.setText(
                    f"min: {ymin:.3f} V\nmax: {ymax:.3f} V\nRMS: {rms:.3f} V\nP2P: {p2p:.3f} V\n"
                    f"Period: {period_time*1000:.1f}ms | Freq: {freq_str}")
            else:
                self.stats_label.setText(
                    f"min: {ymin:.3f} V\nmax: {ymax:.3f} V\nRMS: {rms:.3f} V\nP2P: {p2p:.3f} V")
                
    def _get_triggered_window(self):
        """Get triggered window of data for display - simplified with error handling"""
        try:
            # Check if we have any data at all
            if not hasattr(self, 'voltage_buffer') or self.total_samples_received == 0:
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
                
            # Get enough data for trigger analysis - not the entire time scale!
            # We need enough data to find triggers and extract the requested periods
            if self.trigger_system.detected_period is not None:
                # Get more periods worth of data for analysis - increased for high frequencies
                # For high frequencies, we need more data for reliable trigger detection
                if self.trigger_system.detected_period < 1000:  # High freq (>75Hz at 75kHz sample rate)
                    # For high frequencies, use at least 15 periods of data for better reliability
                    analysis_samples = int(15 * self.trigger_system.detected_period)
                    # But cap at reasonable size
                    analysis_samples = min(analysis_samples, int(0.5 * self.sample_rate))  # Max 0.5 seconds
                    analysis_samples = max(analysis_samples, 2000)  # Minimum 2000 samples for high freq
                elif self.trigger_system.detected_period < 5000:  # Medium freq (15-75Hz)
                    # For medium frequencies, use 10 periods 
                    analysis_samples = int(10 * self.trigger_system.detected_period)
                    analysis_samples = min(analysis_samples, int(1.0 * self.sample_rate))  # Max 1 second
                    analysis_samples = max(analysis_samples, 1500)  # Minimum 1500 samples
                elif self.trigger_system.detected_period > self.sample_rate // 2:  # Period > 0.5 seconds (< 2Hz)
                    # For very low frequencies, use a fixed analysis window
                    analysis_samples = min(int(3 * self.sample_rate), self.total_samples_received)  # Max 3 seconds
                elif self.trigger_system.detected_period > self.sample_rate // 10:  # Period > 0.1 seconds (< 10Hz)
                    # For low frequencies (3-10Hz), ensure we get enough data
                    analysis_samples = max(int(5 * self.trigger_system.detected_period), int(1.5 * self.sample_rate))  # At least 1.5 seconds
                    analysis_samples = min(analysis_samples, int(3 * self.sample_rate))  # Max 3 seconds
                else:
                    # Default case - use 8 periods for better reliability
                    analysis_samples = int(8 * self.trigger_system.detected_period)
                    analysis_samples = min(analysis_samples, int(2 * self.sample_rate))  # Max 2 seconds
            else:
                # Fallback: get more data to help with period detection - increased for high freq support
                analysis_samples = int(1.5 * self.sample_rate)  # Increased from 1.0 seconds
                
            # Ensure we don't request more than available
            available = min(analysis_samples, self.total_samples_received)
            
            # Need minimum data for meaningful trigger analysis - increased for high frequencies
            min_required = max(1000, int(0.01 * self.sample_rate))  # At least 1000 samples or 10ms of data
            if available < min_required:
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
                
            # Update auto level if needed (only once per frame)
            if self.trigger_system.auto_level:
                self.trigger_system.update_auto_level(voltage_data)
                
            # Get triggers - this is the ONLY call to find_triggers per frame
            triggers = self.trigger_system.find_triggers(voltage_data)
            
            # Get triggered window using the simple method with the actual triggers
            triggered_voltage, offset = self.trigger_system.get_triggered_window(voltage_data, triggers, self.trigger_system.periods_to_display)
            
            if len(triggered_voltage) == 0:
                return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
                
            # Create proper time axis for triggered display
            if len(triggered_voltage) > 0:
                # Create a time axis that shows the proper duration
                if self.trigger_system.detected_period is not None:
                    # Calculate the actual time duration we're showing
                    period_time = self.trigger_system.detected_period / self.sample_rate
                    total_time = self.trigger_system.periods_to_display * period_time
                    final_time = np.linspace(0, total_time, len(triggered_voltage))
                else:
                    # Fallback: use sample-based time
                    final_time = np.arange(len(triggered_voltage)) / self.sample_rate
            else:
                final_time = np.array([])
                
            # Downsample if needed
            if len(triggered_voltage) > self.MAX_POINTS_TO_PLOT:
                stride = len(triggered_voltage) // self.MAX_POINTS_TO_PLOT
                triggered_voltage = triggered_voltage[::stride]
                final_time = final_time[::stride]
                
            # Ensure arrays have matching lengths
            min_len = min(len(final_time), len(triggered_voltage))
            final_time = final_time[:min_len]
            triggered_voltage = triggered_voltage[:min_len]
                
            # Store triggers for display update (avoid re-computation)
            self._last_triggers = triggers
            self._last_trigger_data = voltage_data
            self._last_time_data = time_data
                
            return final_time, triggered_voltage
            
        except Exception as e:
            print(f"Error in triggered window: {e}")
            # Return empty arrays on error to prevent crash
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        
    def _update_trigger_display(self, x_data, y_data):
        """Update trigger visualization on the plot - find actual trigger crossings"""
        if len(y_data) == 0:
            return
            
        # Update trigger level line
        self.trigger_line.setPos(self.trigger_system.level)
        
        # Find actual trigger crossings in the displayed waveform
        if len(x_data) > 1 and len(y_data) > 1:
            trigger_times = []
            trigger_voltages = []
            
            # Look for trigger crossings in the displayed data
            level = self.trigger_system.level
            edge = self.trigger_system.edge
            
            # Find crossings based on trigger mode
            if self.trigger_system.mode == 'auto':
                # For auto mode, look for rising edges crossing the level
                for i in range(1, len(y_data)):
                    if y_data[i-1] < level and y_data[i] >= level:
                        # Linear interpolation to find exact crossing point
                        if y_data[i] != y_data[i-1]:
                            frac = (level - y_data[i-1]) / (y_data[i] - y_data[i-1])
                            trigger_time = x_data[i-1] + frac * (x_data[i] - x_data[i-1])
                        else:
                            trigger_time = x_data[i]
                        trigger_times.append(trigger_time)
                        trigger_voltages.append(level)
            else:
                # For normal mode, respect the edge setting
                for i in range(1, len(y_data)):
                    crossing = False
                    if edge == 'rising' and y_data[i-1] < level and y_data[i] >= level:
                        crossing = True
                    elif edge == 'falling' and y_data[i-1] > level and y_data[i] <= level:
                        crossing = True
                    
                    if crossing:
                        # Linear interpolation to find exact crossing point
                        if y_data[i] != y_data[i-1]:
                            frac = (level - y_data[i-1]) / (y_data[i] - y_data[i-1])
                            trigger_time = x_data[i-1] + frac * (x_data[i] - x_data[i-1])
                        else:
                            trigger_time = x_data[i]
                        trigger_times.append(trigger_time)
                        trigger_voltages.append(level)
            
            # Limit to reasonable number of markers for performance
            if len(trigger_times) > 10:
                trigger_times = trigger_times[:10]
                trigger_voltages = trigger_voltages[:10]
            
            if len(trigger_times) > 0:
                self.trigger_markers.setData(trigger_times, trigger_voltages)
            else:
                self.trigger_markers.setData([], [])
                
            # Update status with simplified trigger info (period/frequency now in live stats)
            if self.trigger_system.detected_period is not None:
                status_text = f"Status: {self.trigger_system.mode.title()} | Locked"
                self.trigger_status_label.setText(status_text)
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #90EE90; padding: 5px; }")
            else:
                self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | Searching...")
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")
        else:
            self.trigger_markers.setData([], [])
            if self.trigger_enabled:
                self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | No triggers")
                self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFB6C1; padding: 5px; }")

    def reset_trigger_system(self):
        """Force reset trigger system - clear all state and restart detection"""
        self.trigger_system.reset_trigger_system()
        # Clear any stored trigger data
        self._last_triggers = np.array([])
        self._last_trigger_data = np.array([])
        self._last_time_data = np.array([])
        
        if self.trigger_enabled:
            # Temporarily disable and re-enable to force fresh start
            self.trigger_enabled = False
            self.trigger_enabled = True
            self.trigger_status_label.setText(f"Status: {self.trigger_system.mode.title()} | Reset - Searching...")
            self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFE4B5; padding: 5px; }")
        else:
            self.trigger_status_label.setText("Status: Disabled")
            self.trigger_status_label.setStyleSheet("QLabel { background-color: #f0f0f0; padding: 5px; }")
        self.trigger_markers.setData([], [])
        print("Trigger system force reset - all state cleared")

    def on_dma_error_occurred(self):
        """Handle DMA error notification"""
        # Only reset trigger system if it has been stable for a while
        # This prevents constant resets during normal DMA operation
        import time
        
        # Track reset frequency to prevent excessive resets
        current_time = time.time()
        if not hasattr(self, '_last_dma_reset_time'):
            self._last_dma_reset_time = 0
        if not hasattr(self, '_dma_reset_count'):
            self._dma_reset_count = 0
        
        # If we're resetting too frequently, just ignore this error
        if current_time - self._last_dma_reset_time < 2.0:  # Less than 2 seconds since last reset
            self._dma_reset_count += 1
            if self._dma_reset_count > 3:  # More than 3 resets in 2 seconds
                print("DMA error: Too many recent resets, ignoring")
                return
        else:
            self._dma_reset_count = 0  # Reset counter after 2 seconds
        
        self._last_dma_reset_time = current_time
        
        # Check if trigger system has been working recently
        if (self.trigger_system.last_successful_trigger_time is not None and 
            time.time() - self.trigger_system.last_successful_trigger_time < 5.0):
            # Trigger system was working recently, don't reset completely
            # Just clear the template to allow re-learning but preserve period detection
            self.trigger_system.trigger_template = None
            self.trigger_system.template_update_counter = 0
            print("DMA error: Partial trigger system reset (preserved period detection)")
        else:
            # Complete reset only if trigger system wasn't working anyway
            self.trigger_system.reset_trigger_system()
            print("DMA error: Complete trigger system reset")
        
        self.trigger_status_label.setText("Status: DMA Error")
        self.trigger_status_label.setStyleSheet("QLabel { background-color: #FFB6C1; padding: 5px; }")
        self.trigger_markers.setData([], [])

    def update_plot_y_range(self):
        """Update plot Y-axis range based on current ADC range setting"""
        if self.current_adc_range == 0:
            # 2 Vpp = ±1V
            y_min, y_max = -1.0, 1.0
            range_text = "2 Vpp (±1V)"
        else:
            # 8 Vpp = ±4V
            y_min, y_max = -4.0, 4.0
            range_text = "8 Vpp (±4V)"
        
        self.plot_widget.setYRange(y_min, y_max)
        self.trigger_level_spinbox.setRange(y_min, y_max)
        print(f"Plot Y-axis updated for {range_text}")

    def on_adc_range_button_clicked(self, button):
        """Handle ADC input range button click (0 = 2V, 1 = 8V)"""
        try:
            # Get the button ID from the button group
            idx = self.range_button_group.id(button)
            self.current_adc_range = idx
            self.driver.set_adc_input_range(int(idx))
            self.update_plot_y_range()
            range_text = "2V" if idx == 0 else "8V"
            voltage_range = "±1V" if idx == 0 else "±4V"
            self.status_bar.showMessage(f"ADC range set to {range_text} ({voltage_range}) - Plot auto-scaled")
        except Exception as e:
            self.status_bar.showMessage(f"Failed to set ADC range: {e}")

    def on_filtering_enabled_changed(self, enabled):
        """Handle filtering enable/disable"""
        self.trigger_system.enable_filtering = enabled
        if enabled:
            self.filter_window_spinbox.setEnabled(True)
        else:
            self.filter_window_spinbox.setEnabled(False)

    def on_filter_window_changed(self, value):
        """Handle filtering window size change"""
        self.trigger_system.filter_window = value
        if self.trigger_system.enable_filtering:
            self.filter_window_spinbox.setValue(value)

def main():
    """Main function to run the application."""
    logger.info("Attempting to connect to instrument...")
    try:
        client = connect(DEFAULT_HOST, 'alpha15-laser-control')
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