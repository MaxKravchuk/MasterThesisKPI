import os
from dotenv import load_dotenv
import sys
import numpy as np
import cv2 as cv
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
                             QPlainTextEdit, QRadioButton)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer, Qt
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from sympy.strategies.tree import treeapply
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
        self.proximity_timer = QTimer()
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

        # Top control layout with Start, Stop, and Color Picker
        top_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        top_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        top_layout.addWidget(self.stop_button)

        self.color_selector = QComboBox()
        self.color_selector.addItems(["Red", "Green", "Blue"])
        self.color_selector.currentTextChanged.connect(self.update_selected_color)
        top_layout.addWidget(self.color_selector)

        main_layout.addLayout(top_layout)

        # options_layout = QHBoxLayout()
        #
        # left_options_layout = QHBoxLayout()
        # self.b1 = QRadioButton("Tracking mode")
        # self.b1.setChecked(True)
        # self.b1.toggled.connect(self.toggleCameraMode)
        # options_layout.addWidget(self.b1)
        #
        # self.b2 = QRadioButton("Driving mode")
        # options_layout.addWidget(self.b2)
        # self.b2.toggled.connect(self.toggleCameraMode)
        # options_layout.addLayout(left_options_layout)
        #
        # right_options_layout = QHBoxLayout()
        # self.b3 = QRadioButton("Ground target")
        # self.b3.setChecked(True)
        # options_layout.addWidget(self.b3)
        #
        # self.b4 = QRadioButton("Air target")
        # options_layout.addWidget(self.b4)
        #
        # options_layout.addLayout(right_options_layout)
        # main_layout.addLayout(options_layout)

        # Top Box
        top_box = QLabel()
        top_box.setFixedHeight(25)
        top_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_box.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(top_box)

        # Main video layout with left, right, and main video frames
        video_layout = QHBoxLayout()

        # Left box
        left_box = QLabel()
        left_box.setFixedSize(25, 512)
        left_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_box.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(left_box)

        # Main video display
        self.main_video_label = QLabel("Main frame\n512 x 512")
        self.main_video_label.setFixedSize(512, 512)
        self.main_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_video_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.main_video_label)

        # Right box
        right_box = QLabel()
        right_box.setFixedSize(25, 512)
        right_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_box.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(right_box)

        main_layout.addLayout(video_layout)

        # Bottom Box
        bottom_box = QLabel()
        bottom_box.setFixedHeight(25)
        bottom_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_box.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(bottom_box)

        # Status bar
        self.status_label = QPlainTextEdit()
        self.status_label.setReadOnly(True)
        self.status_label.setPlaceholderText("Scene info")
        self.status_label.setStyleSheet("border: 1px solid black;")
        self.status_label.setFixedHeight(80)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.closeEvent = self.on_close

    def set_noVideo_pixmap(self):
        self.main_video_label.setPixmap(self.no_video_pixmap.scaled(512, 512, Qt.AspectRatioMode.KeepAspectRatio,
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
            self.vision_sensors['Main'] = self.sim.getObject('/PioneerP3DX_main/UR5/Vision_sensor')
            #self.vision_sensors['Front_camera'] = self.sim.getObject('/PioneerP3DX_main/Front_camera')
            self.vision_sensor_script_funcs = self.sim.getScriptFunctions(
                self.sim.getObject('/PioneerP3DX_main/UR5/Vision_sensor/Script'))
            self.proximity_sensor = self.sim.getObject('/PioneerP3DX_main/UR5/Proximity_sensor')
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

                self.proximity_timer.timeout.connect(self.check_proximity_sensor)
                self.proximity_timer.start(5000)
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

    # def toggleCameraMode(self):
    #     try:
    #         if self.b1.isChecked():
    #             self.vision_sensor_script_funcs.toggleTracking(1)
    #         elif self.b2.isChecked():
    #             self.vision_sensor_script_funcs.toggleTracking(0)
    #     except Exception as e:
    #         print(f'An error occurred while toggling camera mode: {e}')

    def get_real_time_video_output(self):
        try:
            # img, resolution = None, None
            # if self.b1.isChecked():
            #     img, resolution = self.sim.getVisionSensorImage(self.vision_sensors['Main'])
            # elif self.b2.isChecked():
            #     img, resolution = self.sim.getVisionSensorImage(self.vision_sensors['Front_camera'])
            img, resolution = self.sim.getVisionSensorImage(self.vision_sensors['Main'])
            if img is not None and resolution:
                frame = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                frame = cv.flip(frame, 0)

                # Process frame with YOLO model
                results = self.model.predict(source=frame, save=False, show=False, conf=0.5, verbose=False)
                annotated_frame = results[0].plot()

                # Convert frame for PyQt display
                annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
                qimg = QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0],
                              annotated_frame.strides[0], QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.main_video_label.setPixmap(pixmap)
        except Exception as e:
            print(f'An error occurred while getting main video output: {e}')

    def check_proximity_sensor(self):
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
