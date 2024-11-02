import sys
import numpy as np
import cv2 as cv
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QComboBox
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from ultralytics import YOLO

class VisionSensorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.client = RemoteAPIClient()
        self.sim = self.client.require('sim')
        self.simulation_running = False
        self.vision_sensor_handle = None
        self.vision_sensor_script_funcs = None
        self.video_timer = QTimer()
        self.model = YOLO('./model_robot.pt')
        self.selected_color = 'Red'
        self.color_ranges = {
            'Red': 'Red',
            'Green':  'Green',
            'Blue': 'Blue'
        }
        self.init_ui()
        self.load_scene()

    def init_ui(self):
        self.setWindowTitle("Vision Sensor Control with YOLO")
        layout = QVBoxLayout()

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_simulation)
        layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button)

        self.color_selector = QComboBox()
        self.color_selector.addItems(["Red", "Green", "Blue"])
        self.color_selector.currentTextChanged.connect(self.update_selected_color)
        layout.addWidget(self.color_selector)

        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        self.setLayout(layout)
        self.closeEvent = self.on_close

    def load_scene(self):
        try:
            print('Loading scene...')
            self.sim.loadScene('E:/KPI/Dipl/pyClient/objectTracking.ttt')
            print('Scene loaded.')
            self.vision_sensor_handle = self.sim.getObject('/UR5/Vision_sensor')
            print(self.vision_sensor_handle)
            self.vision_sensor_script_funcs = self.sim.getScriptFunctions(self.sim.getObject('/UR5/Vision_sensor/Script'))
        except Exception as e:
            print(f'An error occurred while loading the scene: {e}')

    def start_simulation(self):
        if not self.simulation_running:
            try:
                print('Starting the simulation...')
                self.sim.startSimulation()
                self.simulation_running = True
                print('Simulation started.')
                self.video_timer.timeout.connect(self.get_real_time_video_output)
                self.video_timer.start(100)
            except Exception as e:
                print(f'An error occurred while starting the simulation: {e}')

    def stop_simulation(self):
        if self.simulation_running:
            try:
                print('Stopping the simulation...')
                self.sim.stopSimulation()
                self.simulation_running = False
                self.video_timer.stop()
                print('Simulation stopped.')
            except Exception as e:
                print(f'An error occurred while stopping the simulation: {e}')

    def update_selected_color(self):
        self.vision_sensor_script_funcs.setColorToTrack(self.selected_color)

    def get_real_time_video_output(self):
        try:
            img, resolution = self.sim.getVisionSensorImg(self.vision_sensor_handle)
            if img is not None and resolution:
                frame = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
                frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
                frame = cv.flip(frame, 0)

                # Process frame with YOLO model
                results = self.model.predict(source=frame, save=False, show=False, conf=0.5)
                annotated_frame = results[0].plot()

                # Convert frame for PyQt display
                annotated_frame = cv.cvtColor(annotated_frame, cv.COLOR_BGR2RGB)
                qimg = QImage(annotated_frame.data, annotated_frame.shape[1], annotated_frame.shape[0], annotated_frame.strides[0], QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                self.video_label.setPixmap(pixmap)
        except Exception as e:
            print(f'An error occurred while getting video output: {e}')

    def on_close(self, event):
        if self.simulation_running:
            self.stop_simulation()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisionSensorApp()
    window.show()
    sys.exit(app.exec())