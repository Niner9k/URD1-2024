from ultralytics import YOLO
# Specify the dataset path
dataset_path = "model"  # Update if necessary

# Load and train YOLOv8 model
model = YOLO("yolov8s.pt")

# Train the model
model.train(
    data=f"{dataset_path}/data.yaml",
    epochs=25,
    imgsz=800,
    plots=True
)
