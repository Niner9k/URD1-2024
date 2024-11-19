from ultralytics import YOLO

# Specify the dataset path
dataset_path = "model4"  # Update if necessary

# Load the YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model with an increased batch size
model.train(
    data=f"{dataset_path}/data.yaml",
    epochs=25,
    imgsz=800,
    batch=100,  # Set this to your desired batch size
    plots=True
)
