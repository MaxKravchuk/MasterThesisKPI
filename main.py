import os
from dotenv import load_dotenv
import sys
import numpy as np
import cv2 as cv
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox,
                             QPlainTextEdit)
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
        self.proximity_timer = QTimer()
        self.sensor_timer = QTimer()  # Timer for reading ultrasonic sensors
        self.model = YOLO(os.getenv('YOLO_MODEL_PATH'))
        self.no_video_pixmap = QPixmap(os.getenv('NO_VIDEO_IMAGE_PATH'))
        self.selected_color = 'Red'
        self.selected_mode = 'Tracking'
        self.color_ranges = {
            'Red': 'Red',
            'Green': 'Green',
            'Blue': 'Blue'
        }
        self.status_label = None
        self.cameraMode = 1
        self.pressed_keys = set()  # Set to keep track of pressed keys

        # Ultrasonic sensors mapping
        self.sensor_handles = {}
        self.sensor_boxes = {}
        self.sensor_distances = {}

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

        options_layout = QHBoxLayout()

        # Group 1 - Tracking/Driving mode
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Tracking", "Driving"])
        self.mode_selector.currentTextChanged.connect(self.toggleCameraMode)
        options_layout.addWidget(self.mode_selector)

        main_layout.addLayout(options_layout)

        # Top Box
        self.top_box = QLabel()
        self.top_box.setFixedHeight(25)
        self.top_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.top_box.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.top_box)

        # Main video layout with left, right, and main video frames
        video_layout = QHBoxLayout()

        # Left box
        self.left_box = QLabel()
        self.left_box.setFixedSize(25, 512)
        self.left_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_box.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.left_box)

        # Main video display
        self.main_video_label = QLabel("Main frame\n512 x 512")
        self.main_video_label.setFixedSize(512, 512)
        self.main_video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_video_label.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.main_video_label)

        # Right box
        self.right_box = QLabel()
        self.right_box.setFixedSize(25, 512)
        self.right_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_box.setStyleSheet("border: 1px solid black;")
        video_layout.addWidget(self.right_box)

        main_layout.addLayout(video_layout)

        # Bottom Box
        self.bottom_box = QLabel()
        self.bottom_box.setFixedHeight(25)
        self.bottom_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bottom_box.setStyleSheet("border: 1px solid black;")
        main_layout.addWidget(self.bottom_box)

        # Status bar
        self.status_label = QPlainTextEdit()
        self.status_label.setReadOnly(True)
        self.status_label.setPlaceholderText("Scene info")
        self.status_label.setStyleSheet("border: 1px solid black;")
        self.status_label.setFixedHeight(80)
        main_layout.addWidget(self.status_label)

        self.setLayout(main_layout)
        self.closeEvent = self.on_close

        # Set focus policy to accept key events
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFocus()

        # Prevent child widgets from stealing focus
        for widget in self.findChildren(QWidget):
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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
            self.vision_sensors['Front_camera'] = self.sim.getObject('/PioneerP3DX_main/Front_camera')
            self.vision_sensor_script_funcs = self.sim.getScriptFunctions(
                self.sim.getObject('/PioneerP3DX_main/UR5/Vision_sensor/Script'))
            self.proximity_sensor = self.sim.getObject('/PioneerP3DX_main/UR5/Proximity_sensor')
            for name in ['red', 'green', 'blue']:
                self.target_handlers[name] = self.sim.getObject(f"/PioneerP3DX_{name}")

            # Get motor handles
            self.left_motor = self.sim.getObject('/PioneerP3DX_main/leftMotor')
            self.right_motor = self.sim.getObject('/PioneerP3DX_main/rightMotor')

            # Get ultrasonic sensor handles
            sensor_indices = {
                'front': [3, 4],
                'right': [7, 8],
                'back': [11, 12],
                'left': [0, 15]
            }
            for direction, indices in sensor_indices.items():
                self.sensor_handles[direction] = []
                for idx in indices:
                    sensor_name = f'/PioneerP3DX_main/ultrasonicSensor[{idx}]'
                    handle = self.sim.getObject(sensor_name)
                    self.sensor_handles[direction].append(handle)
            self.append_status('Ultrasonic sensor handles retrieved successfully.')

            # Map sensors to UI boxes
            self.sensor_boxes = {
                'front': self.top_box,
                'right': self.right_box,
                'back': self.bottom_box,
                'left': self.left_box
            }

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

                # Sensor timer for reading ultrasonic sensors
                self.sensor_timer.timeout.connect(self.read_ultrasonic_sensors)
                self.sensor_timer.start(100)  # Adjust the interval as needed

                # Set focus to capture key events
                self.setFocus()

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
                self.sensor_timer.stop()
                self.stop_robot()  # Stop the robot when simulation stops
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

    def toggleCameraMode(self):
        if self.simulation_running:
            try:
                self.selected_mode = self.mode_selector.currentText()
                if self.selected_mode == 'Tracking':
                    self.vision_sensor_script_funcs.toggleTracking(1)
                    self.cameraMode = 1
                    self.stop_robot()  # Stop the robot when switching modes
                elif self.selected_mode == 'Driving':
                    self.vision_sensor_script_funcs.toggleTracking(0)
                    self.cameraMode = 0
                # Set focus to capture key events after mode change
                self.setFocus()
            except Exception as e:
                print(f'An error occurred while toggling camera mode: {e}')

    def get_real_time_video_output(self):
        try:
            imgMain, resolutionMain = self.sim.getVisionSensorImg(self.vision_sensors['Main'])
            imgFC, resolutionFC = self.sim.getVisionSensorImg(self.vision_sensors['Front_camera'])
            if imgMain is not None and imgFC is not None and resolutionMain and resolutionFC:

                if self.cameraMode == 1:
                    frame = np.frombuffer(imgMain, dtype=np.uint8).reshape(resolutionMain[1], resolutionMain[0], 3)
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
                elif self.cameraMode == 0:
                    frame = np.frombuffer(imgFC, dtype=np.uint8).reshape(resolutionFC[1], resolutionFC[0], 3)
                    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                    frame = cv.flip(frame, 0)
                    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                    qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                                  frame.strides[0], QImage.Format.Format_RGB888)
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

    def read_ultrasonic_sensors(self):
        if self.simulation_running and self.cameraMode == 0:
            try:
                for direction, handles in self.sensor_handles.items():
                    min_distance = float('inf')
                    for handle in handles:
                        result, distance, detected_point, detected_object_handle, detected_surface_normal_vector = \
                            self.sim.readProximitySensor(handle)
                        if result > 0 and distance < min_distance:
                            min_distance = distance

                    # Store the minimum distance for this direction
                    self.sensor_distances[direction] = min_distance if min_distance != float('inf') else None

                # Update the UI boxes based on the distances
                self.update_ui_boxes()

            except Exception as e:
                print(f'An error occurred while reading ultrasonic sensors: {e}')

    def update_ui_boxes(self):
        # Define thresholds for color changes (in meters)
        thresholds = {
            'green': 1.0,
            'yellow': 0.5,
            'red': 0.2
        }

        for direction, distance in self.sensor_distances.items():
            box = self.sensor_boxes[direction]
            if distance is None:
                # No obstacle detected
                color = 'green'
            elif distance < thresholds['red']:
                color = 'red'
            elif distance < thresholds['yellow']:
                color = 'yellow'
            else:
                color = 'green'

            box.setStyleSheet(f"background-color: {color}; border: 1px solid black;")

    def on_close(self, event):
        if self.simulation_running:
            self.stop_simulation()
        event.accept()

    def keyPressEvent(self, event):
        if self.simulation_running and self.cameraMode == 0:
            key = event.key()
            if key not in self.pressed_keys:
                self.pressed_keys.add(key)
                self.update_robot_movement()

    def keyReleaseEvent(self, event):
        if self.simulation_running and self.cameraMode == 0:
            key = event.key()
            if key in self.pressed_keys:
                self.pressed_keys.remove(key)
                self.update_robot_movement()

    def update_robot_movement(self):
        if self.simulation_running and self.cameraMode == 0:
            forward = Qt.Key.Key_W in self.pressed_keys
            backward = Qt.Key.Key_S in self.pressed_keys
            left = Qt.Key.Key_A in self.pressed_keys
            right = Qt.Key.Key_D in self.pressed_keys

            v_left = 0
            v_right = 0
            velocity = 2.0  # Adjust the speed as needed

            if forward and not backward:
                v_left += velocity
                v_right += velocity
            elif backward and not forward:
                v_left -= velocity
                v_right -= velocity

            if left and not right:
                v_left -= velocity / 2
                v_right += velocity / 2
            elif right and not left:
                v_left += velocity / 2
                v_right -= velocity / 2

            # Limit velocities to max/min values if needed
            max_velocity = 5.0
            v_left = max(min(v_left, max_velocity), -max_velocity)
            v_right = max(min(v_right, max_velocity), -max_velocity)

            # Set the wheel velocities
            self.sim.setJointTargetVelocity(self.left_motor, v_left)
            self.sim.setJointTargetVelocity(self.right_motor, v_right)
        else:
            # If not in driving mode or simulation not running, stop the robot
            self.stop_robot()

    def stop_robot(self):
        # Stop the robot's wheels
        if hasattr(self, 'left_motor') and hasattr(self, 'right_motor'):
            self.sim.setJointTargetVelocity(self.left_motor, 0)
            self.sim.setJointTargetVelocity(self.right_motor, 0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionSensorApp()
    window.show()
    sys.exit(app.exec())
