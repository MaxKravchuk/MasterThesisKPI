from ultralytics import YOLO
import cv2

model = YOLO('model_robot.pt')

cap = cv2.VideoCapture('C:/Users/Flink/Videos/2024-11-02 14-18-37.mkv')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, show=False, conf=0.5)

    annotated_frame = results[0].plot()

    # Display the frame with annotations
    cv2.imshow('YOLOv11 Object Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close display window
cap.release()
cv2.destroyAllWindows()