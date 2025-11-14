"""
Simple GUI for LiNvidia Broadcast noise suppression

Provides an easy-to-use interface for controlling noise suppression
"""

import sys
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QComboBox, QSlider, QGroupBox, QTextEdit
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("Warning: PyQt6 not installed. Install with: pip install PyQt6")

from ..audio import AudioCapture, AudioPlayback
from ..noise_suppression import RealtimeNoiseSuppression
from ..utils import Config


class NoiseSuppressionWorker(QThread):
    """Worker thread for running noise suppression"""

    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        input_device: str,
        output_device: str,
        strength: float,
        parent=None
    ):
        super().__init__(parent)
        self.input_device = input_device
        self.output_device = output_device
        self.strength = strength
        self.noise_suppression = None
        self.is_running = False

    def run(self):
        """Run noise suppression in background thread"""
        try:
            self.status_update.emit("Initializing...")

            # Create config
            config = Config()
            config.noise_suppression.strength = self.strength

            # Create noise suppression
            self.noise_suppression = RealtimeNoiseSuppression(
                config=config,
                input_device=self.input_device,
                output_device=self.output_device
            )

            # Start
            self.status_update.emit("Running...")
            self.noise_suppression.start()
            self.is_running = True

            # Keep thread alive
            while self.is_running:
                self.msleep(100)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.noise_suppression:
                self.noise_suppression.stop()
            self.status_update.emit("Stopped")

    def stop(self):
        """Stop noise suppression"""
        self.is_running = False


class LiNvidiaGUI(QMainWindow):
    """Main GUI window for LiNvidia Broadcast"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("LiNvidia Broadcast - Noise Suppression")
        self.setGeometry(100, 100, 600, 500)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("LiNvidia Broadcast")
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Real-time AI Noise Suppression for Linux")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(20)

        # Device selection group
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout()

        # Input device
        input_layout = QHBoxLayout()
        input_label = QLabel("Input Device:")
        input_label.setMinimumWidth(100)
        self.input_combo = QComboBox()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.input_combo)
        device_layout.addLayout(input_layout)

        # Output device
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Device:")
        output_label.setMinimumWidth(100)
        self.output_combo = QComboBox()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_combo)
        device_layout.addLayout(output_layout)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Load devices
        self.load_devices()

        # Suppression strength
        strength_group = QGroupBox("Noise Suppression Strength")
        strength_layout = QVBoxLayout()

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setMinimum(0)
        self.strength_slider.setMaximum(100)
        self.strength_slider.setValue(95)
        self.strength_slider.valueChanged.connect(self.update_strength_label)

        self.strength_label = QLabel("95%")
        self.strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        strength_layout.addWidget(self.strength_slider)
        strength_layout.addWidget(self.strength_label)

        strength_group.setLayout(strength_layout)
        layout.addWidget(strength_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Noise Suppression")
        self.start_button.setMinimumHeight(40)
        self.start_button.clicked.connect(self.start_suppression)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.clicked.connect(self.stop_suppression)
        self.stop_button.setEnabled(False)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        layout.addLayout(button_layout)

        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(100)
        self.status_text.setText("Ready to start")

        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)

        layout.addWidget(status_group)

        # Add stretch to push everything up
        layout.addStretch()

    def load_devices(self):
        """Load available audio devices"""
        try:
            # Load input devices
            input_devices = AudioCapture.list_devices()
            for dev in input_devices:
                self.input_combo.addItem(dev['name'], dev['index'])

            # Load output devices
            output_devices = AudioPlayback.list_devices()
            for dev in output_devices:
                self.output_combo.addItem(dev['name'], dev['index'])

            self.log_status(f"Loaded {len(input_devices)} input and {len(output_devices)} output devices")

        except Exception as e:
            self.log_status(f"Error loading devices: {e}")

    def update_strength_label(self, value):
        """Update strength label when slider changes"""
        self.strength_label.setText(f"{value}%")

    def start_suppression(self):
        """Start noise suppression"""
        try:
            # Get selected devices
            input_device = self.input_combo.currentText()
            output_device = self.output_combo.currentText()
            strength = self.strength_slider.value() / 100.0

            self.log_status(f"Starting noise suppression...")
            self.log_status(f"Input: {input_device}")
            self.log_status(f"Output: {output_device}")
            self.log_status(f"Strength: {strength:.2f}")

            # Create and start worker
            self.worker = NoiseSuppressionWorker(
                input_device,
                output_device,
                strength
            )

            self.worker.status_update.connect(self.log_status)
            self.worker.error_occurred.connect(self.handle_error)
            self.worker.start()

            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.input_combo.setEnabled(False)
            self.output_combo.setEnabled(False)
            self.strength_slider.setEnabled(False)

        except Exception as e:
            self.log_status(f"Error starting: {e}")

    def stop_suppression(self):
        """Stop noise suppression"""
        if self.worker:
            self.log_status("Stopping...")
            self.worker.stop()
            self.worker.wait()
            self.worker = None

        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.input_combo.setEnabled(True)
        self.output_combo.setEnabled(True)
        self.strength_slider.setEnabled(True)

    def handle_error(self, error_msg):
        """Handle error from worker"""
        self.log_status(f"ERROR: {error_msg}")
        self.stop_suppression()

    def log_status(self, message):
        """Add message to status log"""
        self.status_text.append(message)

    def closeEvent(self, event):
        """Handle window close"""
        if self.worker:
            self.stop_suppression()
        event.accept()


def launch_gui():
    """Launch the GUI application"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required for the GUI")
        print("Install with: pip install PyQt6")
        return 1

    app = QApplication(sys.argv)
    window = LiNvidiaGUI()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(launch_gui())
