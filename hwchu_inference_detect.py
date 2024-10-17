from ultralytics import YOLO
import cv2

model_path = 'yolov8m.pt'
# best_model = YOLO('yolov8m.yaml', task='detect').load(model_path)
best_model = YOLO(model_path)

results = best_model('./ultralytics/assets/bus.jpg')

'''Visualize'''
plots = results[0].plot(show=False)

cv2.imwrite('result_detect.jpg', plots)