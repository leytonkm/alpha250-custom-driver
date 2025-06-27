#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
from koheron import connect, command, ConnectionError

# --- PyQt6 and pyqtgraph Imports ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QTextEdit
)
from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QPalette, QColor
import pyqtgraph as pg
from pyqtgraph import SignalProxy
from collections import deque

# --- Constants ---
APP_VERSION = "1.0.1"
DEFAULT_HOST = os.environ.get('HOST', '192.168.1.20')
DEFAULT_SAMPLE_RATE_HZ = 100_000  # Fixed decimated rate

# --- Koheron Driver Interface ---
class CurrentRamp:
    """A wrapper for the Koheron driver commands."""
    def __init__(self, client):
        self.client = client

    @command()
    def start_adc_streaming(self): pass
    @command()
    def stop_adc_streaming(self): pass
    @command()
    def is_adc_streaming_active(self): return self.client.recv_bool()
    @command()
    def set_cic_decimation_rate(self, rate): pass
    @command()
    def get_decimated_sample_rate(self): return self.client.recv_double()
    @command()
    def get_adc_stream_voltages(self, num_samples):
        return self.client.recv_vector(dtype='float32')

# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # --- Streaming / plotting parameters ---
        self.time_window_s = 5.0  # default visible window in seconds (modifiable by user)
        self.elapsed_time = 0.0   # running time since streaming started

        # Ring buffers to hold incoming data (will be sized once sample_rate is known)
        self.sample_rate = DEFAULT_SAMPLE_RATE_HZ  # will be overwritten upon connection
        self.chunk_size = int(0.05 * self.sample_rate)  # pull 50-ms worth of samples every update
        self.x_buffer = deque()
        self.y_buffer = deque()
        self.total_samples_received = 0
        self.driver = None
        self.data_timer = QTimer(self)
        self.data_timer.setInterval(50)  # 50 ms updates
        self.data_timer.timeout.connect(self.fetch_data)

        self.setWindowTitle(f"Laser ADC Scope v{APP_VERSION}")
        self.setGeometry(100, 100, 1600, 900)
        self.setup_ui()

    def setup_ui(self):
        """Initializes the widgets and layout."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        # --- Control Panel (Left Side) ---
        control_panel = QWidget()
        control_panel.setFixedWidth(300)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Connection Section
        conn_group, conn_layout_parent = self.create_group("Connection")
        conn_layout = QGridLayout()
        conn_layout_parent.addLayout(conn_layout)
        self.conn_status_label = QLabel("Disconnected")
        self.conn_status_label.setStyleSheet("color: #FF6B6B;")
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_device)
        conn_layout.addWidget(self.connect_button, 0, 0)
        conn_layout.addWidget(self.conn_status_label, 0, 1)

        # Streaming Section
        stream_group, stream_layout_parent = self.create_group("Streaming Control")
        stream_layout = QHBoxLayout()
        stream_layout_parent.addLayout(stream_layout)
        self.start_button = QPushButton("Start Streaming")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_streaming)
        self.stop_button = QPushButton("Stop Streaming")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_streaming)
        stream_layout.addWidget(self.start_button)
        stream_layout.addWidget(self.stop_button)

        # Plotting Section
        plot_group, plot_layout_parent = self.create_group("Plotting Controls")
        plot_layout = QGridLayout()
        plot_layout_parent.addLayout(plot_layout)
        plot_layout.addWidget(QLabel("Time Window (s):"), 0, 0)
        self.time_window_spinbox = QDoubleSpinBox()
        self.time_window_spinbox.setRange(0.01, 30.0)
        self.time_window_spinbox.setSingleStep(0.1)
        self.time_window_spinbox.setValue(self.time_window_s)
        self.time_window_spinbox.setDecimals(2)
        self.time_window_spinbox.valueChanged.connect(self.update_time_window)
        plot_layout.addWidget(self.time_window_spinbox, 0, 1)

        self.autoscale_button = QPushButton("Autoscale V")
        self.autoscale_button.clicked.connect(self.autoscale_plot)
        plot_layout.addWidget(self.autoscale_button, 1, 0, 1, 2)
        
        # Debug Section
        debug_group, debug_layout = self.create_group("Debug Log")
        self.debug_log = QTextEdit()
        self.debug_log.setReadOnly(True)
        self.debug_log.setFixedHeight(200)
        self.debug_log.append(f"[{time.strftime('%H:%M:%S')}] Application started. Welcome!")
        self.test_log_button = QPushButton("Test Log")
        self.test_log_button.clicked.connect(self.on_test_log_button_clicked)
        debug_layout.addWidget(self.debug_log)
        debug_layout.addWidget(self.test_log_button)
        
        control_layout.addWidget(debug_group) # Add it first
        control_layout.addWidget(conn_group)
        control_layout.addWidget(stream_group)
        control_layout.addWidget(plot_group)

        # --- Plot (Right Side) ---
        plot_widget = pg.PlotWidget()
        main_layout.addWidget(plot_widget)
        main_layout.addWidget(control_panel)

        self.plot = plot_widget.getPlotItem()
        self.plot.setLabels(left="Voltage (V)", bottom="Time (s)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setYRange(-0.5, 0.5)
        self.plot.setXRange(0, 5)
        self.plot_curve = self.plot.plot(pen=pg.mkPen('#50A7F2', width=2))

        # --- Hover tracker ---
        self.point_marker = pg.ScatterPlotItem(size=8, pen=pg.mkPen(None), brush=pg.mkBrush('#FF3333'))
        self.plot.addItem(self.point_marker)
        # Mouse move proxy (limit to 30 Hz)
        self.mouse_proxy = SignalProxy(self.plot.scene().sigMouseMoved, rateLimit=30, slot=self.mouse_moved)

        # Status Bar
        self.statusBar().showMessage("Ready.")

    def create_group(self, title):
        """Helper to create a styled group box and returns both frame and its layout."""
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout()
        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; padding-top: 5px;")
        layout.addWidget(label)
        frame.setLayout(layout)
        return frame, layout

    def connect_device(self):
        """Initiates a connection attempt."""
        if self.driver:
            return
        try:
            self.log_message(f"Attempting to connect to {DEFAULT_HOST}...")
            client = connect(DEFAULT_HOST, 'currentramp', restart=False)
            self.driver = CurrentRamp(client)
            # Query the actual decimated sample rate from the instrument so that
            # plotting accurately reflects time. Fallback to default if command fails.
            try:
                self.sample_rate = int(self.driver.get_decimated_sample_rate())
            except Exception:
                self.sample_rate = DEFAULT_SAMPLE_RATE_HZ
            self.chunk_size = max(1, int(0.05 * self.sample_rate))
            # Reset buffers sized for at least twice the maximum window to reduce reallocation cost
            self.x_buffer = deque(maxlen=int(self.sample_rate * 2 * 30))  # 30-s safety margin
            self.y_buffer = deque(maxlen=self.x_buffer.maxlen)
            self.conn_status_label.setText("Connected")
            self.conn_status_label.setStyleSheet("color: #2ECC71;")
            self.connect_button.setText("Connected")
            self.connect_button.setEnabled(False)
            self.start_button.setEnabled(True)
            self.log_message(f"Successfully connected to {DEFAULT_HOST}.")
        except (ConnectionError, Exception) as e:
            self.conn_status_label.setText(f"Connection Failed: {e}")
            self.conn_status_label.setStyleSheet("color: #FF6B6B;")
            self.connect_button.setText("Connect")
            self.connect_button.setEnabled(True)
            self.start_button.setEnabled(False)
            self.log_message(f"Could not connect to {DEFAULT_HOST}: {e}")

    def start_streaming(self):
        """Starts the data stream."""
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        if not self.driver:
            self.log_message("Not connected.")
            return
        try:
            self.driver.start_adc_streaming()
            time.sleep(0.3)
            if self.driver.is_adc_streaming_active():
                self.log_message("Streaming started.")
                self.elapsed_time = 0.0
                self.total_samples_received = 0
                self.x_buffer.clear()
                self.y_buffer.clear()
                self.data_timer.start()
        except Exception as e:
            self.log_message(f"Error starting streaming: {e}")
            self.stop_button.setEnabled(False)
            self.start_button.setEnabled(True)

    def stop_streaming(self):
        """Stops the data stream."""
        self.data_timer.stop()
        self.stop_button.setEnabled(False)
        self.start_button.setEnabled(True)
        if self.driver:
            try:
                self.driver.stop_adc_streaming()
                self.log_message("Streaming stopped.")
            except Exception as e:
                self.log_message(f"Error stopping stream: {e}")

    def update_time_window(self, value):
        """Updates the time window for acquisition."""
        self.time_window_s = value
        # Adjust the visible X-range instantly if we are already streaming
        if len(self.x_buffer):
            latest_time = self.x_buffer[-1]
            self.plot.setXRange(max(0.0, latest_time - self.time_window_s), latest_time, padding=0.0)
        # Adjust buffer size if user increases window beyond current capacity
        desired_len = int(self.sample_rate * self.time_window_s * 2)  # double margin
        if self.x_buffer.maxlen is None or self.x_buffer.maxlen < desired_len:
            self.x_buffer = deque(self.x_buffer, maxlen=desired_len)
            self.y_buffer = deque(self.y_buffer, maxlen=desired_len)
        self.log_message(f"Time window set to {value:.2f} s")

    def autoscale_plot(self):
        """Autoscales the plot's Y-axis."""
        self.plot.enableAutoRange(axis='y')

    def on_test_log_button_clicked(self):
        """Callback for the Test Log button."""
        self.log_message("Manual log test successful.")

    def fetch_data(self):
        if not self.driver:
            return
        try:
            voltages = self.driver.get_adc_stream_voltages(self.chunk_size)
            if voltages is None or len(voltages) == 0:
                return

            n_new = len(voltages)
            # Build x-axis based on running sample counter to get absolute time
            new_times = (np.arange(self.total_samples_received,
                                   self.total_samples_received + n_new,
                                   dtype=float) / self.sample_rate)
            self.total_samples_received += n_new

            # Extend ring buffers
            self.x_buffer.extend(new_times)
            self.y_buffer.extend(voltages)

            if len(self.x_buffer) == 0:
                return

            # Determine indices corresponding to visible window
            latest_time = self.x_buffer[-1]
            window_start_time = max(0.0, latest_time - self.time_window_s)

            # Convert deques to numpy for slicing
            x_arr = np.fromiter(self.x_buffer, dtype=float)
            y_arr = np.fromiter(self.y_buffer, dtype=float)

            # Slice for current window
            idx_start = np.searchsorted(x_arr, window_start_time)
            x_visible = x_arr[idx_start:]
            y_visible = y_arr[idx_start:]

            # Update plot
            self.plot_curve.setData(x_visible, y_visible)
            self.plot.setXRange(window_start_time, latest_time, padding=0.0)
        except Exception as e:
            self.log_message(f"Data fetch error: {e}")

    def closeEvent(self, event):
        """Ensures clean shutdown."""
        self.statusBar().showMessage("Shutting down...")
        self.stop_streaming()
        event.accept()

    def log_message(self, message):
        print(message)
        self.statusBar().showMessage(message, 5000)
        if hasattr(self, 'debug_log'):
            self.debug_log.append(f"[{time.strftime('%H:%M:%S')}] {message}")
            self.debug_log.verticalScrollBar().setValue(self.debug_log.verticalScrollBar().maximum())

    def mouse_moved(self, evt):
        pos = evt[0]  # PyQtGraph gives tuple (event)
        if not self.plot.sceneBoundingRect().contains(pos):
            return
        mousePoint = self.plot.vb.mapSceneToView(pos)
        x_pos = mousePoint.x(); y_pos = mousePoint.y()
        xdata, ydata = self.plot_curve.getData()
        if xdata is None or len(xdata)==0:
            self.point_marker.setData([], [])
            return

        idx = np.searchsorted(xdata, x_pos)
        if idx >= len(xdata):
            idx = len(xdata) - 1

        actual_t = xdata[idx]
        actual_v = ydata[idx]
        # Show marker only if cursor is close to the curve (within 5% of full scale)
        if abs(y_pos - actual_v) < 0.05:
            self.point_marker.setData([actual_t], [actual_v])
            self.statusBar().showMessage(f"t = {actual_t:.3f} s   V = {actual_v:.3f} V")
        else:
            self.point_marker.setData([], [])
            self.statusBar().clearMessage()

def set_dark_theme(app):
    """A simple dark theme for the application."""
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    app.setStyleSheet("""
        QToolTip {
            color: #ffffff; background-color: #2a82da; border: 1px solid white;
        }
        QWidget {
            font-size: 14px;
        }
        QPushButton {
            border: 1px solid #555;
            padding: 5px;
            border-radius: 3px;
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #555555, stop: 1 #444444);
        }
        QPushButton:hover {
            background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                                              stop: 0 #666666, stop: 1 #555555);
        }
        QPushButton:pressed {
            background-color: #333333;
        }
        QPushButton:disabled {
            color: #888;
            background-color: #454545;
        }
        QComboBox, QDoubleSpinBox {
            border: 1px solid #555;
            padding: 3px;
            border-radius: 3px;
        }
        QFrame[frameShape="5"] { /* QFrame.StyledPanel */
            border: 1px solid #444;
            border-radius: 5px;
            margin-top: 1ex;
        }
    """)

# --- Main Execution ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    set_dark_theme(app)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())