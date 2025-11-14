"""
Full-featured GUI for LiNvidia Broadcast with audio and video

Provides comprehensive control over noise suppression and background effects
"""

import sys
from typing import Optional

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QComboBox, QSlider, QGroupBox, QTextEdit,
        QTabWidget, QFileDialog, QRadioButton, QButtonGroup, QCheckBox
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal
    from PyQt6.QtGui import QFont
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False

from ..audio import AudioCapture, AudioPlayback
from ..noise_suppression import RealtimeNoiseSuppression
from ..background_effects import RealtimeBackgroundEffects, BackgroundEffect
from ..tracking import FramingMode
from ..utils import Config


class VideoEffectsWorker(QThread):
    """Worker thread for running background effects"""

    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        camera_id: int,
        effect: BackgroundEffect,
        blur_strength: int,
        background_image: Optional[str],
        output_type: str,
        virtual_device: Optional[str],
        enable_auto_frame: bool = False,
        framing_mode: str = 'center',
        parent=None
    ):
        super().__init__(parent)
        self.camera_id = camera_id
        self.effect = effect
        self.blur_strength = blur_strength
        self.background_image = background_image
        self.output_type = output_type
        self.virtual_device = virtual_device
        self.enable_auto_frame = enable_auto_frame
        self.framing_mode = framing_mode
        self.bg_effects = None
        self.is_running = False

    def run(self):
        """Run background effects in background thread"""
        try:
            self.status_update.emit("Initializing video...")

            # Create background effects
            self.bg_effects = RealtimeBackgroundEffects(
                camera_id=self.camera_id,
                output_type=self.output_type,
                virtual_device=self.virtual_device,
                effect=self.effect,
                blur_strength=self.blur_strength,
                enable_auto_frame=self.enable_auto_frame,
                framing_mode=self.framing_mode
            )

            # Load background image if provided
            if self.background_image and self.effect == BackgroundEffect.REPLACE:
                self.bg_effects.set_background_image(self.background_image)

            # Start
            self.status_update.emit("Running...")
            self.bg_effects.start()
            self.is_running = True

            # Keep thread alive
            while self.is_running:
                # Process frames
                frame = self.bg_effects.capture.read(timeout=1.0)
                if frame is not None:
                    processed = self.bg_effects.process_frame(frame)
                    self.bg_effects.display.show(processed)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            if self.bg_effects:
                self.bg_effects.stop()
            self.status_update.emit("Stopped")

    def stop(self):
        """Stop background effects"""
        self.is_running = False

    def update_effect(self, effect: BackgroundEffect):
        """Update effect while running"""
        if self.bg_effects:
            self.bg_effects.set_effect(effect)

    def update_blur_strength(self, strength: int):
        """Update blur strength while running"""
        if self.bg_effects:
            self.bg_effects.set_blur_strength(strength)


class LiNvidiaFullGUI(QMainWindow):
    """
    Full-featured GUI for LiNvidia Broadcast

    Supports both audio noise suppression and video background effects
    """

    def __init__(self):
        super().__init__()
        self.audio_worker = None
        self.video_worker = None
        self.background_image_path = None
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("LiNvidia Broadcast - AI Effects for Linux")
        self.setGeometry(100, 100, 700, 700)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Title
        title = QLabel("LiNvidia Broadcast")
        title_font = QFont()
        title_font.setPointSize(20)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Real-time AI Effects for Linux")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        # Tab widget for audio and video
        tabs = QTabWidget()
        tabs.addTab(self.create_audio_tab(), "🎤 Noise Suppression")
        tabs.addTab(self.create_video_tab(), "📹 Background Effects")
        layout.addWidget(tabs)

        # Status display
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()

        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(120)
        self.status_text.setText("Ready to start")

        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)

        layout.addWidget(status_group)

    def create_audio_tab(self) -> QWidget:
        """Create audio controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Device selection
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout()

        # Input device
        input_layout = QHBoxLayout()
        input_label = QLabel("Input:")
        input_label.setMinimumWidth(80)
        self.audio_input_combo = QComboBox()
        input_layout.addWidget(input_label)
        input_layout.addWidget(self.audio_input_combo)
        device_layout.addLayout(input_layout)

        # Output device
        output_layout = QHBoxLayout()
        output_label = QLabel("Output:")
        output_label.setMinimumWidth(80)
        self.audio_output_combo = QComboBox()
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.audio_output_combo)
        device_layout.addLayout(output_layout)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Load audio devices
        self.load_audio_devices()

        # Suppression strength
        strength_group = QGroupBox("Noise Suppression Strength")
        strength_layout = QVBoxLayout()

        self.audio_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.audio_strength_slider.setMinimum(0)
        self.audio_strength_slider.setMaximum(100)
        self.audio_strength_slider.setValue(95)
        self.audio_strength_slider.valueChanged.connect(self.update_audio_strength_label)

        self.audio_strength_label = QLabel("95%")
        self.audio_strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        strength_layout.addWidget(self.audio_strength_slider)
        strength_layout.addWidget(self.audio_strength_label)
        strength_group.setLayout(strength_layout)
        layout.addWidget(strength_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.audio_start_button = QPushButton("Start Noise Suppression")
        self.audio_start_button.setMinimumHeight(40)
        self.audio_start_button.clicked.connect(self.start_audio)

        self.audio_stop_button = QPushButton("Stop")
        self.audio_stop_button.setMinimumHeight(40)
        self.audio_stop_button.clicked.connect(self.stop_audio)
        self.audio_stop_button.setEnabled(False)

        button_layout.addWidget(self.audio_start_button)
        button_layout.addWidget(self.audio_stop_button)
        layout.addLayout(button_layout)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_video_tab(self) -> QWidget:
        """Create video controls tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Camera selection
        camera_group = QGroupBox("Camera")
        camera_layout = QHBoxLayout()
        camera_label = QLabel("Camera:")
        camera_label.setMinimumWidth(80)
        self.camera_combo = QComboBox()
        self.load_cameras()
        camera_layout.addWidget(camera_label)
        camera_layout.addWidget(self.camera_combo)
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Background effect selection
        effect_group = QGroupBox("Background Effect")
        effect_layout = QVBoxLayout()

        self.effect_button_group = QButtonGroup()

        self.effect_none_radio = QRadioButton("None (Original)")
        self.effect_blur_radio = QRadioButton("Blur Background")
        self.effect_remove_radio = QRadioButton("Remove Background (Green Screen)")
        self.effect_replace_radio = QRadioButton("Replace Background")

        self.effect_blur_radio.setChecked(True)

        self.effect_button_group.addButton(self.effect_none_radio, 0)
        self.effect_button_group.addButton(self.effect_blur_radio, 1)
        self.effect_button_group.addButton(self.effect_remove_radio, 2)
        self.effect_button_group.addButton(self.effect_replace_radio, 3)

        effect_layout.addWidget(self.effect_none_radio)
        effect_layout.addWidget(self.effect_blur_radio)
        effect_layout.addWidget(self.effect_remove_radio)
        effect_layout.addWidget(self.effect_replace_radio)

        # Background image selection
        bg_image_layout = QHBoxLayout()
        self.bg_image_button = QPushButton("Select Background Image...")
        self.bg_image_button.clicked.connect(self.select_background_image)
        self.bg_image_label = QLabel("No image selected")
        bg_image_layout.addWidget(self.bg_image_button)
        bg_image_layout.addWidget(self.bg_image_label)
        effect_layout.addLayout(bg_image_layout)

        effect_group.setLayout(effect_layout)
        layout.addWidget(effect_group)

        # Blur strength
        blur_group = QGroupBox("Blur Strength")
        blur_layout = QVBoxLayout()

        self.blur_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.blur_strength_slider.setMinimum(1)
        self.blur_strength_slider.setMaximum(100)
        self.blur_strength_slider.setValue(25)
        self.blur_strength_slider.valueChanged.connect(self.update_blur_strength_label)
        self.blur_strength_slider.valueChanged.connect(self.update_video_blur_strength)

        self.blur_strength_label = QLabel("25")
        self.blur_strength_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        blur_layout.addWidget(self.blur_strength_slider)
        blur_layout.addWidget(self.blur_strength_label)
        blur_group.setLayout(blur_layout)
        layout.addWidget(blur_group)

        # Auto-framing controls
        autoframe_group = QGroupBox("Auto-Framing")
        autoframe_layout = QVBoxLayout()

        # Enable auto-framing checkbox
        self.autoframe_enable_checkbox = QCheckBox("Enable Auto-Framing")
        self.autoframe_enable_checkbox.setChecked(False)
        self.autoframe_enable_checkbox.stateChanged.connect(self.update_autoframe_controls)
        autoframe_layout.addWidget(self.autoframe_enable_checkbox)

        # Framing mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Framing Mode:")
        mode_label.setMinimumWidth(100)
        self.framing_mode_combo = QComboBox()
        self.framing_mode_combo.addItem("Center", "center")
        self.framing_mode_combo.addItem("Headroom (Professional)", "headroom")
        self.framing_mode_combo.addItem("Tight (Close-up)", "tight")
        self.framing_mode_combo.addItem("Wide (Show Context)", "wide")
        self.framing_mode_combo.addItem("Group (Multiple People)", "group")
        self.framing_mode_combo.setEnabled(False)
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.framing_mode_combo)
        autoframe_layout.addLayout(mode_layout)

        # Info label
        self.autoframe_info_label = QLabel("Auto-framing keeps you centered and properly framed")
        self.autoframe_info_label.setStyleSheet("color: gray; font-style: italic;")
        autoframe_layout.addWidget(self.autoframe_info_label)

        autoframe_group.setLayout(autoframe_layout)
        layout.addWidget(autoframe_group)

        # Output selection
        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout()

        self.output_window_radio = QRadioButton("Display Window")
        self.output_virtual_radio = QRadioButton("Virtual Camera (v4l2loopback)")

        self.output_window_radio.setChecked(True)

        output_layout.addWidget(self.output_window_radio)
        output_layout.addWidget(self.output_virtual_radio)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # Control buttons
        button_layout = QHBoxLayout()

        self.video_start_button = QPushButton("Start Background Effects")
        self.video_start_button.setMinimumHeight(40)
        self.video_start_button.clicked.connect(self.start_video)

        self.video_stop_button = QPushButton("Stop")
        self.video_stop_button.setMinimumHeight(40)
        self.video_stop_button.clicked.connect(self.stop_video)
        self.video_stop_button.setEnabled(False)

        button_layout.addWidget(self.video_start_button)
        button_layout.addWidget(self.video_stop_button)
        layout.addLayout(button_layout)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def load_audio_devices(self):
        """Load audio devices"""
        try:
            # Load input devices
            input_devices = AudioCapture.list_devices()
            for dev in input_devices:
                self.audio_input_combo.addItem(dev['name'], dev['index'])

            # Load output devices
            output_devices = AudioPlayback.list_devices()
            for dev in output_devices:
                self.audio_output_combo.addItem(dev['name'], dev['index'])

        except Exception as e:
            self.log_status(f"Error loading audio devices: {e}")

    def load_cameras(self):
        """Load available cameras"""
        from ..video import VideoCapture
        try:
            devices = VideoCapture.list_devices()
            for dev in devices:
                self.camera_combo.addItem(dev['name'], dev['id'])

            if not devices:
                self.camera_combo.addItem("No cameras found", -1)

        except Exception as e:
            self.log_status(f"Error loading cameras: {e}")

    def update_audio_strength_label(self, value):
        """Update audio strength label"""
        self.audio_strength_label.setText(f"{value}%")

    def update_blur_strength_label(self, value):
        """Update blur strength label"""
        self.blur_strength_label.setText(str(value))

    def update_video_blur_strength(self, value):
        """Update video blur strength while running"""
        if self.video_worker:
            self.video_worker.update_blur_strength(value)

    def update_autoframe_controls(self, state):
        """Update auto-framing controls based on checkbox state"""
        enabled = (state == Qt.CheckState.Checked.value)
        self.framing_mode_combo.setEnabled(enabled)

    def select_background_image(self):
        """Select background replacement image"""
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_name:
            self.background_image_path = file_name
            import os
            self.bg_image_label.setText(os.path.basename(file_name))
            self.log_status(f"Background image selected: {file_name}")

    def get_selected_effect(self) -> BackgroundEffect:
        """Get selected background effect"""
        button_id = self.effect_button_group.checkedId()
        effects = [
            BackgroundEffect.NONE,
            BackgroundEffect.BLUR,
            BackgroundEffect.REMOVE,
            BackgroundEffect.REPLACE
        ]
        return effects[button_id]

    def start_audio(self):
        """Start audio noise suppression"""
        try:
            input_device = self.audio_input_combo.currentText()
            output_device = self.audio_output_combo.currentText()
            strength = self.audio_strength_slider.value() / 100.0

            self.log_status("Starting noise suppression...")
            self.log_status(f"Input: {input_device}")
            self.log_status(f"Output: {output_device}")
            self.log_status(f"Strength: {strength:.2f}")

            # Import here to avoid circular dependency
            from ..noise_suppression import RealtimeNoiseSuppressionWorker

            self.audio_worker = RealtimeNoiseSuppressionWorker(
                input_device,
                output_device,
                strength
            )

            self.audio_worker.status_update.connect(self.log_status)
            self.audio_worker.error_occurred.connect(self.handle_error)
            self.audio_worker.start()

            # Update UI
            self.audio_start_button.setEnabled(False)
            self.audio_stop_button.setEnabled(True)

        except Exception as e:
            self.log_status(f"Error starting audio: {e}")

    def stop_audio(self):
        """Stop audio noise suppression"""
        if self.audio_worker:
            self.log_status("Stopping audio...")
            self.audio_worker.stop()
            self.audio_worker.wait()
            self.audio_worker = None

        # Update UI
        self.audio_start_button.setEnabled(True)
        self.audio_stop_button.setEnabled(False)

    def start_video(self):
        """Start video background effects"""
        try:
            camera_id = self.camera_combo.currentData()
            if camera_id == -1:
                self.log_status("Error: No camera available")
                return

            effect = self.get_selected_effect()
            blur_strength = self.blur_strength_slider.value()
            output_type = 'window' if self.output_window_radio.isChecked() else 'virtual'
            virtual_device = '/dev/video2' if output_type == 'virtual' else None
            enable_auto_frame = self.autoframe_enable_checkbox.isChecked()
            framing_mode = self.framing_mode_combo.currentData()

            self.log_status("Starting background effects...")
            self.log_status(f"Camera: {self.camera_combo.currentText()}")
            self.log_status(f"Effect: {effect.value}")
            self.log_status(f"Output: {output_type}")
            if enable_auto_frame:
                self.log_status(f"Auto-framing: {framing_mode}")

            # Create worker
            self.video_worker = VideoEffectsWorker(
                camera_id,
                effect,
                blur_strength,
                self.background_image_path,
                output_type,
                virtual_device,
                enable_auto_frame,
                framing_mode
            )

            self.video_worker.status_update.connect(self.log_status)
            self.video_worker.error_occurred.connect(self.handle_error)
            self.video_worker.start()

            # Update UI
            self.video_start_button.setEnabled(False)
            self.video_stop_button.setEnabled(True)

        except Exception as e:
            self.log_status(f"Error starting video: {e}")

    def stop_video(self):
        """Stop video background effects"""
        if self.video_worker:
            self.log_status("Stopping video...")
            self.video_worker.stop()
            self.video_worker.wait()
            self.video_worker = None

        # Update UI
        self.video_start_button.setEnabled(True)
        self.video_stop_button.setEnabled(False)

    def handle_error(self, error_msg):
        """Handle error from worker"""
        self.log_status(f"ERROR: {error_msg}")
        self.stop_audio()
        self.stop_video()

    def log_status(self, message):
        """Add message to status log"""
        self.status_text.append(message)

    def closeEvent(self, event):
        """Handle window close"""
        self.stop_audio()
        self.stop_video()
        event.accept()


def launch_full_gui():
    """Launch the full-featured GUI"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt6 is required for the GUI")
        print("Install with: pip install PyQt6")
        return 1

    app = QApplication(sys.argv)
    window = LiNvidiaFullGUI()
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(launch_full_gui())
