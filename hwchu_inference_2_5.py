from ultralytics import YOLO
import cv2

model_path = 'yolov8m.pt'
best_model = YOLO("yolov8m_2_5.yaml", task="detect_2_5").load(model_path)

print('[FOR Inference]')
results = best_model('./ultralytics/assets/bus.jpg')

print('[results[0].boxes.data.shape]')
print(results[0].boxes.data.shape)

'''결과 파일로 저장'''
with open("result_2_5.txt", "w") as file:
    file.write(str(results[0].boxes))

'''Visualize'''
plots = results[0].plot(show=False)
cv2.imwrite('result_2_5.jpg', plots)