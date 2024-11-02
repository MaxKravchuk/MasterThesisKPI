# import keyboard
# import time
# import numpy as np
# import cv2 as cv
# from coppeliasim_zmqremoteapi_client import RemoteAPIClient
#
# client = RemoteAPIClient()
# sim = client.require('sim')
#
# def connect_and_start_simulation():
#     try:
#         print('Loading scene...')
#         sim.loadScene('C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/scenes/vision/objectTracking.ttt')
#         print('Scene loaded.')
#
#         vision_sensor_handle = sim.getObject('/UR5/Vision_sensor')
#         def get_real_time_video_output(vision_sensor_handle_v):
#             while True:
#                 # Get the image data from the vision sensor
#                 img, resolution = sim.getVisionSensorImg(vision_sensor_handle_v)
#                 if img is not None and resolution:
#                     # Convert the image data to a numpy array and reshape it
#                     frame = np.frombuffer(img, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
#                     frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  # Convert to BGR for OpenCV display
#                     frame = cv.flip(frame, 0)  # Flip the frame vertically
#                     cv.imshow('Real-time Vision Sensor Output', frame)
#
#                     # Break on pressing 'q'
#                     if cv.waitKey(1) & keyboard.is_pressed('q'):
#                         break
#
#         while True:
#             if keyboard.is_pressed('s'):
#                 print('Starting the simulation...')
#                 sim.startSimulation()
#                 print('Simulation started.')
#                 get_real_time_video_output(vision_sensor_handle)  # Start displaying video output
#                 time.sleep(0.5)
#             elif keyboard.is_pressed('e'):
#                 print('Stopping the simulation...')
#                 cv.destroyAllWindows()
#                 sim.stopSimulation()
#                 print('Simulation stopped.')
#                 time.sleep(0.5)
#             time.sleep(0.1)
#
#     except Exception as e:
#         print(f'An error occurred: {e}')
#     finally:
#         sim.stopSimulation()
#         cv.destroyAllWindows()
#
#
# if __name__ == "__main__":
#     connect_and_start_simulation()
import numpy as np
import cv2 as cv
import glob

image_files = glob.glob('E:/KPI/Dipl/Photos/*.png')
print(image_files)

