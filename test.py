import torch
from ultralytics import YOLO
# model = YOLO("/home/tthanas/URD1-2024/runs/detect/train/weights/best.pt")
model = YOLO("runs/detect/train5/weights/best.pt")
# result = model(source='Pics/blackmen.jpg', show=True, conf=0.4, save =True)
# result = model(source='Pics/kid1.jpg', show=True, conf=0.4, save =True)
# result = model(source='Pics/kid2.jpeg', show=True, conf=0.4, save =True)
# result = model(source='Pics/oreo.png', show=True, conf=0.4, save =True)
result = model(source='Pics/satwik.jpg', show=True, conf=0.4, save =True)