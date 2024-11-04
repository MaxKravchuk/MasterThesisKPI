from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model on your dipl
model.train(
    data='./dataset.yaml',
    epochs=50,
    imgsz=640,
    name='model'
)

# Save the trained model weights
model.save('model_robot.pt')