from ultralytics import YOLO
import cv2

model_path = 'yolov8n.pt'
model = YOLO("yolov8n_2_5.yaml", task="detect_2_5").load(model_path)

results = model.train(epochs=100, batch=8)