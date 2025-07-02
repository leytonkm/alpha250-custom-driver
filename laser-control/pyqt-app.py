#!/usr/bin/env python3

import sys
import os
import time
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from koheron import connect, command

# --- Constants ---
APP_TITLE = "Laser Control - Live ADC Scope"
DEFAULT_HOST = os.environ.get('HOST', '192.168.1.20')
INSTRUMENT_NAME = 'laser-control'

# Performance settings
UPDATE_INTERVAL_MS = 50  # Update plot every 50 ms (20 FPS)
SAMPLES_PER_UPDATE = 5000 # Fetch 50 ms of data per update (5000 samples @ 100 kHz)

# --- Driver Interface ---
class CurrentRamp:
    """Python interface for the laser-control driver"""
    def __init__(self, client):
        self.client = client

    @command()
    def start_adc_streaming(self): pass
    
    @command()
    def stop_adc_streaming(self): pass

    @command()
    def get_buffer_position(self):
        return self.client.recv_uint32()

    @command()
    def read_adc_buffer_chunk(self, offset, size):
        return self.client.recv_vector(dtype='float32')

    @command()
    def get_decimated_sample_rate(self):
        return self.client.recv_double()

    @command()
    def get_ramp_frequency(self):
        return self.client.recv_double()

# --- Data Acquisition Thread ---

class DataWorker(QtCore.QThread):
    """
    Worker thread for acquiring ADC data from the instrument.
    Decouples data acquisition from the GUI to prevent freezing.
    Can run indefinitely (live stream) or for a fixed number of samples.
    """
    data_ready = QtCore.pyqtSignal(np.ndarray)
    status_changed = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    progress_updated = QtCore.pyqtSignal(int) # For fixed-duration runs

    def __init__(self, driver, samples_to_collect=None):
        super(DataWorker, self).__init__()
        self.driver = driver
        self.running = False
        self.total_buffer_size = 0
        self.read_ptr = 0
        self.samples_to_collect = samples_to_collect
        self.samples_collected = 0

    def run(self):
        """Main acquisition loop."""
        try:
            self.status_changed.emit("Worker thread started")
            self.running = True

            n_desc = 64
            n_pts = 64 * 1024
            self.total_buffer_size = n_desc * n_pts

            self.driver.start_adc_streaming()
            
            # --- Robust warm-up: Actively read and discard initial data ---
            self.status_changed.emit("ADC warming up...")
            warmup_end_time = time.time() + 0.5
            while time.time() < warmup_end_time and self.running:
                write_ptr = self.driver.get_buffer_position()
                samples_available = (write_ptr - self.read_ptr + self.total_buffer_size) % self.total_buffer_size
                if samples_available > 0:
                    read_size = min(samples_available, SAMPLES_PER_UPDATE)
                    # Read data to advance our pointer, but do nothing with the returned data
                    _ = self.driver.read_adc_buffer_chunk(self.read_ptr, read_size)
                    self.read_ptr = (self.read_ptr + read_size) % self.total_buffer_size
                time.sleep(0.02) # Don't hog the CPU while waiting
            
            self.status_changed.emit(f"Starting acquisition at sample index {self.read_ptr}")

            while self.running:
                loop_start_time = time.time()
                
                write_ptr = self.driver.get_buffer_position()
                samples_available = (write_ptr - self.read_ptr + self.total_buffer_size) % self.total_buffer_size

                if samples_available > 0:
                    read_size = min(samples_available, SAMPLES_PER_UPDATE)
                    
                    if self.samples_to_collect is not None:
                        remaining = self.samples_to_collect - self.samples_collected
                        if remaining <= 0:
                            self.running = False
                            continue
                        read_size = min(read_size, remaining)

                    if read_size > 0:
                        data = np.array(self.driver.read_adc_buffer_chunk(self.read_ptr, read_size), dtype=np.float32)
                        
                        # FIX: If we only get zeros, it means the DMA buffer hasn't received real data yet.
                        # Discard it and wait for the next chunk.
                        if data.size > 0 and np.all(data == 0):
                            self.read_ptr = (self.read_ptr + read_size) % self.total_buffer_size
                            self.status_changed.emit("Waiting for valid data from FPGA...")
                            continue # Skip to next loop iteration

                        if data.size > 0:
                            self.data_ready.emit(data)
                            self.read_ptr = (self.read_ptr + data.size) % self.total_buffer_size
                            self.samples_collected += data.size
                            
                            if self.samples_to_collect is not None:
                                progress = int(100 * self.samples_collected / self.samples_to_collect)
                                self.progress_updated.emit(progress)

                elapsed_ms = (time.time() - loop_start_time) * 1000
                sleep_ms = max(0, UPDATE_INTERVAL_MS - elapsed_ms)
                time.sleep(sleep_ms / 1000)

        except Exception as e:
            self.status_changed.emit(f"Error in worker: {e}")
        finally:
            if self.running: # Ensure streaming is stopped if loop breaks unexpectedly
                self.driver.stop_adc_streaming()
            self.status_changed.emit("ADC streaming stopped.")
            self.finished.emit()

    def stop(self):
        """Request the thread to stop."""
        self.running = False

# --- Main Application Window ---

class MainWindow(QtWidgets.QMainWindow):
    """Main GUI window."""
    MAX_POINTS_TO_PLOT = 10000 # Performance: max points to send to setData at once

    def __init__(self, driver):
        super(MainWindow, self).__init__()
        self.driver = driver
        self.data_worker = None
        self.worker_action_pending = None # Used for safe worker transitions

        # --- Data Buffers ---
        self.time_scale_s = 5.0 # Default time window to display
        self.sample_rate = self.driver.get_decimated_sample_rate()
        
        self.max_plot_points = int(10.0 * self.sample_rate)
        self.time_buffer = np.zeros(self.max_plot_points)
        self.voltage_buffer = np.zeros(self.max_plot_points)
        self.buffer_ptr = 0
        self.last_time = 0.0
        self.total_samples_received = 0
        self.is_fixed_run = False
        self.run_duration = 0.0
        self.crosshair_locked = False

        # --- Triggering State ---
        self.is_trigger_enabled = False
        self.trigger_state = "Idle"  # Idle, Armed, Triggered
        self.trigger_level = 0.1  # In Volts
        self.trigger_edge_rising = True
        self.last_sample_for_trigger = 0.0

        self.setup_ui()
        self.connect_signals()

        self.set_time_scale(self.time_scale_s)
        self.status_bar.showMessage("Ready. Connect to an instrument and start streaming.")

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
        self.streaming_group_box = QtWidgets.QGroupBox("Live Streaming")
        streaming_layout = QtWidgets.QVBoxLayout()
        self.start_button = QtWidgets.QPushButton("Start Streaming")
        self.start_button.setCheckable(True)
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
        self.streaming_group_box.setLayout(streaming_layout)

        # --- Fixed Run Controls ---
        self.run_group_box = QtWidgets.QGroupBox("Fixed-Duration Run")
        run_layout = QtWidgets.QGridLayout()
        run_layout.addWidget(QtWidgets.QLabel("Duration:"), 0, 0)
        self.duration_spinbox = QtWidgets.QDoubleSpinBox()
        self.duration_spinbox.setRange(0.000001, 10.0)
        self.duration_spinbox.setDecimals(6)
        self.duration_spinbox.setSingleStep(0.1)
        self.duration_spinbox.setSuffix(" s")
        self.duration_spinbox.setValue(1.0)
        run_layout.addWidget(self.duration_spinbox, 0, 1)
        self.run_button = QtWidgets.QPushButton("Start Run")
        run_layout.addWidget(self.run_button, 1, 0, 1, 2)
        self.run_progress_bar = QtWidgets.QProgressBar()
        self.run_progress_bar.setTextVisible(False)
        run_layout.addWidget(self.run_progress_bar, 2, 0, 1, 2)
        self.run_group_box.setLayout(run_layout)

        # --- Triggering Controls ---
        trigger_group_box = QtWidgets.QGroupBox("Triggering")
        trigger_layout = QtWidgets.QGridLayout()
        self.trigger_enable_checkbox = QtWidgets.QCheckBox("Enable Trigger")
        trigger_layout.addWidget(self.trigger_enable_checkbox, 0, 0, 1, 2)
        trigger_layout.addWidget(QtWidgets.QLabel("Level (mV):"), 1, 0)
        self.trigger_level_spinbox = QtWidgets.QDoubleSpinBox()
        self.trigger_level_spinbox.setRange(-2500, 2500)
        self.trigger_level_spinbox.setDecimals(1)
        self.trigger_level_spinbox.setSingleStep(10)
        self.trigger_level_spinbox.setValue(100.0)
        trigger_layout.addWidget(self.trigger_level_spinbox, 1, 1)
        self.trigger_edge_rising_radio = QtWidgets.QRadioButton("Rising")
        self.trigger_edge_rising_radio.setChecked(True)
        self.trigger_edge_falling_radio = QtWidgets.QRadioButton("Falling")
        trigger_layout.addWidget(self.trigger_edge_rising_radio, 2, 0)
        trigger_layout.addWidget(self.trigger_edge_falling_radio, 2, 1)
        self.arm_trigger_button = QtWidgets.QPushButton("Arm Single Shot")
        trigger_layout.addWidget(self.arm_trigger_button, 3, 0, 1, 2)
        self.trigger_status_label = QtWidgets.QLabel("Status: Idle")
        self.trigger_status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        trigger_layout.addWidget(self.trigger_status_label, 4, 0, 1, 2)
        
        # Add test trigger button for debugging
        self.test_trigger_button = QtWidgets.QPushButton("Test Trigger")
        self.test_trigger_button.setEnabled(False)
        trigger_layout.addWidget(self.test_trigger_button, 5, 0, 1, 2)
        
        trigger_group_box.setLayout(trigger_layout)

        # --- Locked Coordinate Display ---
        locked_coord_group = QtWidgets.QGroupBox("Locked Coordinate")
        locked_coord_layout = QtWidgets.QVBoxLayout()
        self.locked_coord_label = QtWidgets.QLabel("Time: --\nVoltage: --")
        self.locked_coord_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        locked_coord_layout.addWidget(self.locked_coord_label)
        locked_coord_group.setLayout(locked_coord_layout)

        control_layout.addWidget(self.streaming_group_box)
        control_layout.addWidget(self.run_group_box)
        control_layout.addWidget(trigger_group_box)
        control_layout.addWidget(locked_coord_group)
        control_layout.addStretch()

        # --- Plotting Area ---
        self.plot_widget = pg.PlotWidget()
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2))
        self.plot_curve.setDownsampling(auto=True)
        self.plot_widget.setClipToView(True)
        self.plot_widget.setLabel('bottom', "Time", units='s')
        self.plot_widget.setLabel('left', "Voltage", units='V')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(-0.5, 0.5)

        # --- Desmos-style Hover Elements ---
        self.hover_marker = pg.ScatterPlotItem([], [], pxMode=True, size=10, pen=pg.mkPen('w'), brush=pg.mkBrush(255, 255, 255, 120))
        self.locked_marker = pg.ScatterPlotItem([], [], pxMode=True, size=12, pen=pg.mkPen('w'), brush=pg.mkBrush('b'))
        self.coord_text = pg.TextItem(text='', color=(200, 200, 200), anchor=(-0.1, 1.2))
        
        self.plot_widget.addItem(self.hover_marker)
        self.plot_widget.addItem(self.locked_marker)
        self.plot_widget.addItem(self.coord_text)
        
        main_layout.addWidget(self.plot_widget)
        main_layout.addWidget(control_panel)

        # --- Status Bar ---
        self.status_bar = self.statusBar()

    def connect_signals(self):
        """Connect UI signals to slots."""
        self.start_button.toggled.connect(self.toggle_streaming)
        self.run_button.clicked.connect(self.start_fixed_run)
        for val, radio in self.ts_radios.items():
            radio.toggled.connect(lambda checked, v=val: self.set_time_scale(v) if checked else None)

        # Connect trigger signals
        self.trigger_enable_checkbox.toggled.connect(self.set_trigger_enabled)
        self.arm_trigger_button.clicked.connect(self.arm_trigger)
        self.test_trigger_button.clicked.connect(self.test_trigger)
        self.trigger_level_spinbox.valueChanged.connect(lambda val: setattr(self, 'trigger_level', val / 1000.0))
        self.trigger_edge_rising_radio.toggled.connect(lambda checked: setattr(self, 'trigger_edge_rising', checked))

        # Connect plot interaction signals
        self.plot_widget.scene().sigMouseMoved.connect(self.update_crosshair)
        self.plot_widget.scene().sigMouseClicked.connect(self.plot_clicked)

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

    @QtCore.pyqtSlot(bool)
    def toggle_streaming(self, checked):
        if checked:
            self.start_streaming()
            self.start_button.setText("Stop Streaming")
        else:
            self.stop_streaming()
            self.start_button.setText("Start Streaming")

    def start_streaming(self):
        """Start the data acquisition worker for live streaming."""
        self.is_fixed_run = False
        self.run_button.setEnabled(False)
        self.duration_spinbox.setEnabled(False)
        self.start_worker(samples_to_collect=None)

    def stop_streaming(self):
        """Stop the data acquisition worker."""
        if self.data_worker and self.data_worker.isRunning():
            self.worker_action_pending = None # Cancel any pending action
            self.data_worker.stop()
    
    def start_fixed_run(self):
        """Start the data acquisition worker for a fixed duration."""
        self.is_fixed_run = True
        self.run_duration = self.duration_spinbox.value()
        samples_to_collect = int(self.run_duration * self.sample_rate)
        
        self.start_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.duration_spinbox.setEnabled(False)
        self.ts_group_box.setEnabled(False)
        
        self.plot_widget.setXRange(0, self.run_duration)
        self.run_progress_bar.setValue(0)
        
        self.start_worker(samples_to_collect=samples_to_collect)

    def start_worker(self, samples_to_collect=None):
        """Safely stops any existing worker and starts a new one."""
        if self.data_worker and self.data_worker.isRunning():
            # If a worker is running, request it to stop and set a pending action
            # to start a new worker once the old one has finished.
            self.worker_action_pending = lambda: self.start_worker(samples_to_collect)
            self.data_worker.stop()
        else:
            # No worker running, so we can start a new one immediately.
            self.reset_and_clear_buffers()
            self.data_worker = DataWorker(self.driver, samples_to_collect)
            self.connect_worker_signals()
            if samples_to_collect:
                self.data_worker.progress_updated.connect(self.run_progress_bar.setValue)
            self.data_worker.start()

    def reset_and_clear_buffers(self):
        self.buffer_ptr = 0
        self.last_time = 0.0
        self.total_samples_received = 0
        self.voltage_buffer.fill(0)
        self.time_buffer.fill(0)
        self.plot_curve.setData(x=[], y=[])
        self.last_sample_for_trigger = 0.0 # FIX: Reset trigger state
        
        # Reset markers and labels
        self.crosshair_locked = False
        self.hover_marker.setData([], [])
        self.locked_marker.setData([], [])
        self.coord_text.setText('')
        self.locked_coord_label.setText("Time: --\nVoltage: --")

    def connect_worker_signals(self):
        self.data_worker.data_ready.connect(self.update_plot)
        self.data_worker.status_changed.connect(self.status_bar.showMessage)
        self.data_worker.finished.connect(self.on_worker_finished)

    def on_worker_finished(self):
        """Called when either streaming is stopped or a run completes."""
        self.start_button.setEnabled(True)
        self.start_button.setChecked(False)
        self.run_button.setEnabled(True)
        self.duration_spinbox.setEnabled(True)
        self.ts_group_box.setEnabled(True)

        if self.is_fixed_run:
            self.status_bar.showMessage(f"Run of {self.run_duration}s finished. {self.total_samples_received} samples collected.")
            self.run_progress_bar.setValue(100)
        else:
             self.status_bar.showMessage("Streaming stopped.")
        
        # If there's a pending action (like starting another worker), execute it now.
        if self.worker_action_pending:
            action = self.worker_action_pending
            self.worker_action_pending = None
            action()
    
    @QtCore.pyqtSlot(float)
    @QtCore.pyqtSlot(float)
    def set_time_scale(self, scale_s):
        """Update the visible time window."""
        self.time_scale_s = scale_s
        self.plot_curve.setData(x=[], y=[])
        self.status_bar.showMessage(f"Time scale set to {scale_s}s")

    @QtCore.pyqtSlot(np.ndarray)
    def update_plot(self, new_data):
        """Append new data to the circular buffer and update the plot."""
        n_new = new_data.size
        if n_new == 0:
            return

        new_times = self.last_time + (np.arange(n_new) + 1) / self.sample_rate
        self.last_time = new_times[-1]

        # --- Triggering Logic ---
        if self.is_trigger_enabled and self.trigger_state == "Armed":
            # Search for trigger in the new data chunk
            trigger_index_in_chunk = self.find_trigger_point(new_data)

            # Debug: Print signal statistics every few updates when armed
            if self.total_samples_received % 5000 == 0:  # Every ~50ms worth of data
                signal_min, signal_max = np.min(new_data), np.max(new_data)
                signal_mean = np.mean(new_data)
                print(f"📊 Signal stats: range=[{signal_min*1000:.1f}, {signal_max*1000:.1f}]mV, mean={signal_mean*1000:.1f}mV, trigger={self.trigger_level*1000:.1f}mV")

            if trigger_index_in_chunk != -1:
                print(f"🎯 TRIGGER FOUND! Index {trigger_index_in_chunk} in chunk of {len(new_data)} samples")
                self.trigger_state = "Triggered"
                self.update_trigger_status_label()
                
                # Update button to allow re-arming
                self.arm_trigger_button.setText("Re-Arm Trigger")
                self.arm_trigger_button.setEnabled(True)
                
                self.status_bar.showMessage(f"Triggered at T={self.last_time - (n_new - trigger_index_in_chunk) / self.sample_rate:.4f}s!")

                # First, buffer all the new data so we have it available
                start_idx = self.buffer_ptr
                indices = np.arange(start_idx, start_idx + n_new) % self.max_plot_points
                self.voltage_buffer[indices] = new_data
                self.time_buffer[indices] = new_times
                self.buffer_ptr = (start_idx + n_new) % self.max_plot_points
                self.total_samples_received += n_new

                # --- New, Simplified Capture Window Logic ---
                # Get the ramp frequency directly from the instrument.
                ramp_freq = self.driver.get_ramp_frequency()
                # Use a small threshold to avoid division by zero or near-zero frequencies.
                if ramp_freq > 0.01:
                    period_s = 1.0 / ramp_freq
                    # Capture 1.5 periods to show one full cycle plus context.
                    samples_to_capture = int(period_s * 1.5 * self.sample_rate)
                else:
                    # Fallback if frequency is zero (e.g., ramp not running)
                    samples_to_capture = int(0.2 * self.sample_rate) # 200ms

                print(f"📊 Ramp freq={ramp_freq:.2f}Hz. Capturing {samples_to_capture} samples (1.5 periods).")
                
                # Limit capture size to the samples actually available in the buffer
                available_samples = min(self.total_samples_received, self.max_plot_points)
                samples_to_capture = min(samples_to_capture, available_samples)

                # Always capture the most recent data (ending at current buffer position)
                end_pos = self.buffer_ptr
                start_pos = (end_pos - samples_to_capture + self.max_plot_points) % self.max_plot_points
                
                # Extract data, handling buffer wraparound
                if start_pos < end_pos:
                    # Simple case - no wraparound
                    x_data = self.time_buffer[start_pos:end_pos].copy()
                    y_data = self.voltage_buffer[start_pos:end_pos].copy()
                else:
                    # Handle wraparound
                    x_data = np.concatenate((self.time_buffer[start_pos:], self.time_buffer[:end_pos]))
                    y_data = np.concatenate((self.voltage_buffer[start_pos:], self.voltage_buffer[:end_pos]))
                
                if x_data.size > 0:
                    # Re-base the time axis so the first point is t=0 for clarity
                    x_data_rel = x_data - x_data[0]

                    # Set data and fix plot zoom
                    self.plot_curve.setData(x=x_data_rel, y=y_data)
                    self.plot_widget.getPlotItem().setXRange(0, x_data_rel[-1], padding=0.05)
                    self.plot_widget.getPlotItem().enableAutoRange(axis='y')

                self.last_sample_for_trigger = new_data[-1]
                
                # Stop streaming after trigger capture
                if self.data_worker:
                    self.data_worker.stop()
                
                return

        # For triggered mode, if we're armed but haven't triggered yet, just buffer data but don't plot
        if self.is_trigger_enabled and self.trigger_state == "Armed":
            # Buffer the data but don't update plot - we're waiting for trigger
            start_idx = self.buffer_ptr
            indices = np.arange(start_idx, start_idx + n_new) % self.max_plot_points
            self.voltage_buffer[indices] = new_data
            self.time_buffer[indices] = new_times
            self.buffer_ptr = (start_idx + n_new) % self.max_plot_points
            self.total_samples_received += n_new
            self.last_sample_for_trigger = new_data[-1]
            return

        # For triggered mode, if we're triggered, don't update plot (it's frozen)
        if self.is_trigger_enabled and self.trigger_state == "Triggered":
            return

        # Normal live streaming or fixed run mode
        # Buffer the data for non-triggered updates
        start_idx = self.buffer_ptr
        indices = np.arange(start_idx, start_idx + n_new) % self.max_plot_points
        self.voltage_buffer[indices] = new_data
        self.time_buffer[indices] = new_times
        
        self.buffer_ptr = (start_idx + n_new) % self.max_plot_points
        self.total_samples_received += n_new
        self.last_sample_for_trigger = new_data[-1]

        # Update plot for live streaming or fixed run
        if self.is_fixed_run:
            x_data = self.time_buffer[:self.total_samples_received]
            y_data = self.voltage_buffer[:self.total_samples_received]
            self.plot_widget.getPlotItem().enableAutoRange(axis='y')
        else:
            points_to_show = int(self.time_scale_s * self.sample_rate)
            if self.total_samples_received < points_to_show:
                x_data = self.time_buffer[:self.total_samples_received]
                y_data = self.voltage_buffer[:self.total_samples_received]
            else:
                end_ptr = self.buffer_ptr
                start_ptr = (end_ptr - points_to_show + self.max_plot_points) % self.max_plot_points
                if start_ptr < end_ptr:
                    x_data = self.time_buffer[start_ptr:end_ptr]
                    y_data = self.voltage_buffer[start_ptr:end_ptr]
                else:
                    x_data = np.concatenate((self.time_buffer[start_ptr:], self.time_buffer[:end_ptr]))
                    y_data = np.concatenate((self.voltage_buffer[start_ptr:], self.voltage_buffer[:end_ptr]))
            
        if x_data.size > self.MAX_POINTS_TO_PLOT:
            stride = x_data.size // self.MAX_POINTS_TO_PLOT
            self.plot_curve.setData(x=x_data[::stride], y=y_data[::stride])
        else:
            self.plot_curve.setData(x=x_data, y=y_data)

    def set_trigger_enabled(self, enabled):
        """Enable or disable the trigger mode."""
        self.is_trigger_enabled = enabled
        
        # Enable/disable trigger controls based on checkbox state
        self.trigger_level_spinbox.setEnabled(enabled)
        self.trigger_edge_rising_radio.setEnabled(enabled)
        self.trigger_edge_falling_radio.setEnabled(enabled)
        self.arm_trigger_button.setEnabled(enabled)
        self.test_trigger_button.setEnabled(enabled)
        
        if enabled:
            # When enabling trigger mode, disable other streaming controls to avoid conflicts
            self.streaming_group_box.setEnabled(False)
            self.run_group_box.setEnabled(False)
            self.trigger_state = "Idle"
            self.arm_trigger_button.setText("Arm Single Shot")
            self.status_bar.showMessage("Trigger mode enabled. Click 'Arm Single Shot' to start.")
        else:
            # When disabling trigger mode, stop any active streaming and re-enable other controls
            if self.data_worker:
                self.data_worker.stop()
            self.streaming_group_box.setEnabled(True)
            self.run_group_box.setEnabled(True)
            self.trigger_state = "Idle"
            self.plot_widget.getPlotItem().enableAutoRange(axis='y')
            self.status_bar.showMessage("Trigger mode disabled.")
        
        self.update_trigger_status_label()

    def arm_trigger(self):
        """Arm the trigger to wait for an event."""
        if not self.is_trigger_enabled:
            self.status_bar.showMessage("Enable trigger mode first.")
            return

        if self.trigger_state == "Armed":
            self.arm_trigger_button.setText("Arming...")
            self.arm_trigger_button.setEnabled(False)
            # Re-arming: stop the current worker, and _actually_arm_trigger will be called
            # by on_worker_finished via a pending action.
            self.worker_action_pending = self._actually_arm_trigger
            if self.data_worker and self.data_worker.isRunning():
                self.data_worker.stop()
            return
        
        self._actually_arm_trigger()

    def _actually_arm_trigger(self):
        """Internal method to actually arm the trigger."""
        # Update UI state
        self.arm_trigger_button.setText("Armed... (click to re-arm)")
        self.arm_trigger_button.setEnabled(True)
        self.trigger_state = "Armed"
        self.update_trigger_status_label()
        
        # Start streaming to wait for trigger
        self.start_worker()
        
        # Disable autoranging while we wait for a specific event
        self.plot_widget.getPlotItem().disableAutoRange()
        
        # Debug info
        print(f"🎯 Trigger armed: {'Rising' if self.trigger_edge_rising else 'Falling'} edge at {self.trigger_level*1000:.1f}mV")
        
        self.status_bar.showMessage(f"Armed and waiting for {'rising' if self.trigger_edge_rising else 'falling'} edge at {self.trigger_level*1000:.1f}mV...")

    def update_trigger_status_label(self):
        self.trigger_status_label.setText(f"Status: {self.trigger_state}")

    def find_trigger_point(self, y_data):
        """Find the index of the trigger point in the data."""
        if len(y_data) < 2:  # Need at least 2 points
            return -1
        
        # Debug: Check signal range vs trigger level
        signal_min, signal_max = np.min(y_data), np.max(y_data)
        
        # If trigger level is completely outside signal range, we'll never trigger
        if (self.trigger_level < signal_min - 0.01) or (self.trigger_level > signal_max + 0.01):
            # This can be spammy if the signal is temporarily flat, so let's not show it on status bar
            # print(f"Trigger level {self.trigger_level*1000:.1f}mV outside signal range [{signal_min*1000:.1f}, {signal_max*1000:.1f}]mV")
            return -1
        
        # Start search from the first sample, comparing with the last known sample
        prev_sample = self.last_sample_for_trigger
        
        for i in range(len(y_data)):
            current_sample = y_data[i]
            
            # Simple level crossing detection - no slope requirement initially
            trigger_found = False
            
            if self.trigger_edge_rising:
                # Rising edge: signal crosses trigger level going upward
                if prev_sample < self.trigger_level and current_sample >= self.trigger_level:
                    trigger_found = True
            else:
                # Falling edge: signal crosses trigger level going downward
                if prev_sample > self.trigger_level and current_sample <= self.trigger_level:
                    trigger_found = True
            
            if trigger_found:
                # Optional: Add slope check for more robust triggering (but make it much more lenient)
                if i > 0:
                    slope = y_data[i] - y_data[i-1]
                else:
                    slope = current_sample - prev_sample
                
                # Very lenient slope threshold - just ensure we're going in the right direction
                slope_threshold = 0.0001  # Much smaller threshold
                
                if self.trigger_edge_rising and slope >= -slope_threshold:  # Allow small negative slopes
                    return i
                elif not self.trigger_edge_rising and slope <= slope_threshold:  # Allow small positive slopes
                    return i
                # If slope check fails, continue looking for another crossing
            
            prev_sample = current_sample
        
        return -1

    def test_trigger(self):
        """Test trigger functionality by manually firing a trigger."""
        if not self.is_trigger_enabled:
            self.status_bar.showMessage("Enable trigger mode first.")
            return
            
        if self.trigger_state != "Armed":
            self.status_bar.showMessage("Arm the trigger first, then click Test Trigger.")
            return
        
        print("🧪 Manual trigger test activated")
        
        # Simulate finding a trigger by manually calling the trigger logic
        if self.data_worker and self.total_samples_received > 1000:
            # Get some recent data from the buffer for the capture
            end_pos = self.buffer_ptr
            start_pos = (end_pos - 1000 + self.max_plot_points) % self.max_plot_points
            
            if start_pos < end_pos:
                test_data = self.voltage_buffer[start_pos:end_pos].copy()
                test_times = self.time_buffer[start_pos:end_pos].copy()
            else:
                test_data = np.concatenate((self.voltage_buffer[start_pos:], self.voltage_buffer[:end_pos]))
                test_times = np.concatenate((self.time_buffer[start_pos:], self.time_buffer[:end_pos]))
            
            # Manually trigger the capture logic
            self.trigger_state = "Triggered"
            self.update_trigger_status_label()
            self.arm_trigger_button.setText("Re-Arm Trigger")
            self.arm_trigger_button.setEnabled(True)
            
            # Capture a reasonable window of data
            samples_to_capture = min(len(test_data), int(self.sample_rate * 0.2))  # 200ms
            
            x_data = test_times[-samples_to_capture:]
            y_data = test_data[-samples_to_capture:]
            
            self.plot_curve.setData(x=x_data, y=y_data)
            self.plot_widget.getPlotItem().autoRange()
            
            if self.data_worker:
                self.data_worker.stop()
            
            self.status_bar.showMessage("Test trigger fired! Captured current data.")
            print("🧪 Test trigger completed successfully")
        else:
            self.status_bar.showMessage("Not enough data for test trigger. Wait a moment after arming.")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.data_worker:
            self.data_worker.stop()
            self.data_worker.wait(2000) # Wait up to 2 seconds for the thread to finish
        event.accept()

def main():
    """Main function to run the application."""
    print("Attempting to connect to instrument...")
    try:
        client = connect(DEFAULT_HOST, 'laser-control')
        driver = CurrentRamp(client)
        print(f"Connected to {INSTRUMENT_NAME} at {DEFAULT_HOST}")
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Please ensure the instrument is running and accessible. You are likely connected to the wrong network")
        return 1

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    main_win = MainWindow(driver)
    main_win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()