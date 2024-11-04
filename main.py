import os
from dotenv import load_dotenv
import sys
import numpy as np
import cv2 as cv
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
                             QGridLayout, QPlainTextEdit)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ultralytics import YOLO


class VisionSensorApp(QWidget):
    def __init__(self):
        super().__init__()

        # Load environment variables
        load_dotenv()

        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simulation_running = False
        self.vision_sensors = {}
        self.target_handlers = {}
        self.selected_target = None
        self.proximity_sensor = None
        self.vision_sensor_script_funcs = None
        self.video_timer = QTimer()
        self.model = YOLO(os.getenv('YOLO_MODEL_PATH'))
        self.no_video_pixmap = QPixmap(os.getenv('NO_VIDEO_IMAGE_PATH'))
        self.selected_color = 'Red'
        self.color_ranges = {
            'Red': 'Red',
            'Green': 'Green',
            'Blue': 'Blue'
        }
        self.status_label = None
        self.init_ui()
        self.load_scene()

    def init_ui(self):
        self.setWindowTitle("Vision Sensor Control System")

        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        video_grid_layout = QGridLayout()
        scene_info_layout = QHBoxLayout()

        # Header with system name and buttons
        system_name_label = QLabel("Object Tracking System")
        control_layout.addWidget(system_name_label)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        control_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        control_layout.addWidget(self.stop_button)

        self.color_selector = QComboBox()
        self.color_selector.addItems(["Red", "Green", "Blue"])
        self.color_selector.currentTextChanged.connect(self.update_selected_color)
        control_layout.addWidget(self.color_selector)

        main_layout.addLayout(control_layout)

        # Main video display occupying 2x2 grid cells
        self.main_video_label = QLabel("Main video from vision sensor")
        self.main_video_label.setStyleSheet("border: 1px solid black;")
        self.main_video_label.setFixedSize(518, 518)
        self.main_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_grid_layout.addWidget(self.main_video_label, 0, 0, 2, 2)

        # Sub videos display in the third column
        self.vision_sensor_labels = {}
        for i, name in enumerate(["HSV", "Mask"]):
            label = QLabel(f"{name}")
            label.setStyleSheet("border: 1px solid black;")
            label.setFixedSize(256, 256)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vision_sensor_labels[name] = label
            video_grid_layout.addWidget(label, i, 2)

        main_layout.addLayout(video_grid_layout)

        # Status bar
        self.status_label = QPlainTextEdit()
        self.status_label.setReadOnly(True)
        self.status_label.setPlaceholderText("Scene info")
        self.status_label.setStyleSheet("border: 1px solid black;")
        self.status_label.setFixedSize(774, 80)
        scene_info_layout.addWidget(self.status_label)

        main_layout.addLayout(scene_info_layout)

        self.setLayout(main_layout)

        self.closeEvent = self.on_close

    def set_noVideo_pixmap(self):
        self.main_video_label.setPixmap(self.no_video_pixmap.scaled(518, 518, Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation))
        for label in self.vision_sensor_labels.values():
            label.setPixmap(self.no_video_pixmap.scaled(256, 256, Qt.AspectRatioMode.KeepAspectRatio,
                                                        Qt.TransformationMode.SmoothTransformation))

    def append_status(self, text):
        self.status_label.appendPlainText(text)

    def load_scene(self):
        try:
            self.append_status('Loading the scene...')
            self.sim.loadScene(os.getenv('SCENE_PATH'))
            self.append_status('Scene loaded successfully.')

            # Get vision sensor handles
            self.append_status('Getting handles...')
            self.vision_sensors['Main'] = self.sim.getObject('/UR5/Vision_sensor')
            self.vision_sensor_script_funcs = self.sim.getScriptFunctions(self.sim.getObject('/UR5/Vision_sensor/Script'))
            self.vision_sensors['HSV'] = self.sim.getObject('/UR5/hsv')
            self.vision_sensors['Mask'] = self.sim.getObject('/UR5/mask')
            self.proximity_sensor = self.sim.getObject('/UR5/Proximity_sensor')
            for name in ['red', 'green', 'blue']:
                self.target_handlers[name] = self.sim.getObject(f"/PioneerP3DX_{name}")
            self.append_status('Handles retrieved successfully.')

            # Default target
            self.selected_target = self.target_handlers['red']
            self.set_noVideo_pixmap()

        except Exception as e:
            print(f'An error occurred while loading the scene: {e}')

    def start_simulation(self):
        if not self.simulation_running:
            try:
                self.append_status('Starting the simulation...')
                self.sim.startSimulation()
                self.simulation_running = True
                self.append_status('Simulation started.')
                self.video_timer.timeout.connect(self.get_real_time_video_output)
                self.video_timer.start(100)
            except Exception as e:
                print(f'An error occurred while starting the simulation: {e}')

    def stop_simulation(self):
        if self.simulation_running:
            try:
                self.append_status('Stopping the simulation...')
                self.sim.stopSimulation()
                self.simulation_running = False
                self.set_noVideo_pixmap()
                self.video_timer.stop()
                self.append_status('Simulation stopped.')
            except Exception as e:
                print(f'An error occurred while stopping the simulation: {e}')

    def update_selected_color(self):
        if self.simulation_running:
            try:
                self.selected_color = self.color_selector.currentText()
                self.vision_sensor_script_funcs.setColorToTrack(self.selected_color)
                self.selected_target = self.target_handlers[self.selected_color.lower()]
            except Exception as e:
                print(f'An error occurred while updating the selected color: {e}')

    def get_real_time_video_output(self):
        try:
            img, resolution = self.sim.getVisionSensorImg(self.vision_sensors['Main'])
            if img is not None and resolution:
                frame = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                frame = cv.flip(frame, 0)

                # Process frame with YOLO model
                results = self.model.predict(source=frame, save=False, show=False, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()

                # Convert frame for PyQt display
                annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
                qimg = QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0], annotated_frame.strides[0], QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.main_video_label.setPixmap(pixmap)
        except Exception as e:
            print(f'An error occurred while getting main video output: {e}')

        # Process other vision sensors
        for name, handle in self.vision_sensors.items():
            if name != 'Main':
                try:
                    img, resolution = self.sim.getVisionSensorImg(handle)
                    if img is not None and resolution:
                        frame = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
                        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                        frame = cv.flip(frame, 0)

                        # Resize the frame to fit the QLabel
                        frame = cv.resize(frame, (
                        self.vision_sensor_labels[name].width(), self.vision_sensor_labels[name].height()))

                        # Convert frame for PyQt display
                        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0],
                                      QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        self.vision_sensor_labels[name].setPixmap(pixmap)

                except Exception as e:
                    print(f'An error occurred while getting video output for {name}: {e}')

        try:
            res, dist, point, obj, n = self.sim.checkProximitySensor(self.proximity_sensor, self.selected_target)
            if res:
                self.append_status(f"Distance to {obj} is {dist}")
        except Exception as e:
            print(f'An error occurred while reading proximity sensor: {e}')

    def on_close(self, event):
        if self.simulation_running:
            self.stop_simulation()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionSensorApp()
    window.show()
    sys.exit(app.exec())