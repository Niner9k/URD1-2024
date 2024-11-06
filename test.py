import torch
from ultralytics import YOLO
from IPython.display import display, Image
model = YOLO("yolov8n.pt")
result = model(source='/home/tthanas/URD1-2024/Pics/balackmen.jpg', show=True, conf=0.4, save =True)