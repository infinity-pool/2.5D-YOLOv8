from ultralytics import YOLO
import cv2

model_path = 'yolov8m.pt'
model = YOLO("yolov8m_2_5.yaml", task="detect_2_5").load(model_path)
device = 'cpu'

results = model.train(epochs=5, device=device, data="coco8.yaml")

# print('[results[0].boxes.data.shape]')
# print(results[0].boxes.data.shape)

# '''결과 파일로 저장'''
# with open("result_2_5.txt", "w") as file:
#     file.write(str(results[0].boxes))

# '''Visualize'''
# plots = results[0].plot(show=False)
# cv2.imwrite('result_2_5.jpg', plots)