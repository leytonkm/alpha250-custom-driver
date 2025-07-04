#!/usr/bin/env python3

import sys
import os
import time
import numpy as np
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
                    if self.debug_counter >= 20:  # every ~1 s (20 × 50 ms)
                        self.debug_counter = 0
                        desc = self.driver.get_current_descriptor_index()
                        dma_run = self.driver.get_dma_running()
                        dma_idle = self.driver.get_dma_idle()
                        dma_err = self.driver.get_dma_error()
                        logger.debug(
                            "LiveWorker RD=%d WR=%d avail=%d desc=%d running=%s idle=%s err=%s",
                            self.read_ptr, write_ptr, samples_available, desc, dma_run, dma_idle, dma_err)
                        # Auto-restart DMA if it went idle (wrap reached)
                        if dma_idle and not dma_err:
                            logger.warning("DMA idle detected – restarting stream (wraparound)")
                            self.driver.stop_adc_streaming()
                            time.sleep(0.001)
                            self.driver.start_adc_streaming()
                            # Re-align read pointer safe distance behind new write pointer
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
        self.ts_group_box = QtWidgets.QGroupBox("Time Scale")
        ts_layout = QtWidgets.QVBoxLayout()
        self.ts_radios = {}
        for val in [1.0, 2.0, 5.0, 10.0]:
            self.ts_radios[val] = QtWidgets.QRadioButton(f"{val} s")
            ts_layout.addWidget(self.ts_radios[val])
        self.ts_radios[self.time_scale_s].setChecked(True)
        self.ts_group_box.setLayout(ts_layout)
        streaming_layout.addWidget(self.start_button)
        # Freeze checkbox
        self.freeze_checkbox = QtWidgets.QCheckBox("Freeze")
        streaming_layout.addWidget(self.freeze_checkbox)
        streaming_layout.addWidget(self.ts_group_box)
        streaming_group_box.setLayout(streaming_layout)

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

        # --- Desmos-style Hover Elements (live plot only) ---
        self.hover_marker = pg.ScatterPlotItem([], [], pxMode=True, size=10, pen=pg.mkPen('w'), brush=pg.mkBrush(255, 255, 255, 120))
        self.locked_marker = pg.ScatterPlotItem([], [], pxMode=True, size=12, pen=pg.mkPen('w'), brush=pg.mkBrush('b'))
        self.coord_text = pg.TextItem(text='', color=(200, 200, 200), anchor=(-0.1, 1.2))
        
        self.plot_widget.addItem(self.hover_marker)
        self.plot_widget.addItem(self.locked_marker)
        self.plot_widget.addItem(self.coord_text)
        
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
        if self.freeze_checkbox.isChecked():
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
        if self.freeze_checkbox.isChecked():
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

        x_data, y_data = self._get_decimated_window()
        self.plot_widget.setXRange(0, self.time_scale_s, padding=0)
        self.plot_curve.setData(x=x_data, y=y_data)

        if y_data.size > 0:
            ymin = float(np.min(y_data))
            ymax = float(np.max(y_data))
            rms  = float(np.sqrt(np.mean(np.square(y_data))))
            p2p  = ymax - ymin
            self.stats_label.setText(
                f"min: {ymin:.3f} V\nmax: {ymax:.3f} V\nRMS: {rms:.3f} V\nP2P: {p2p:.3f} V")

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