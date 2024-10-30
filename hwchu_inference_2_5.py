from ultralytics import YOLO
import cv2

model_path = 'yolov8m.pt'
# best_model = YOLO("yolov8m_2_5.yaml", task="detect_2_5").load(model_path) # TEST on detect pt model

best_model = YOLO('./runs/detect_2_5/train10/weights/best.pt')

print('- [FOR Inference]')
# results = best_model('./ultralytics/assets/bus.jpg')
results = best_model('../datasets/nuScenes_2_5/images/val/n008-2018-08-30-15-31-50-0400__CAM_BACK__1535657649037558.jpg')

# print('- [results[0].boxes]')
# print(results[0].boxes)

print('- [results[0].boxes.data.shape]')
print(results[0].boxes.data.shape)
print(results[0].boxes.data)

print('- [results[0].dists.data.shape]')
print(results[0].dists.data.shape)
print(results[0].dists.data)

'''결과 파일로 저장'''
with open("result_2_5.txt", "w") as file:
    file.write(str(results[0].boxes))

'''Visualize'''
plots = results[0].plot(show=False)
cv2.imwrite('result_2_5.jpg', plots)

print('<<hwchu_inference_2_5 FINISH!>>')