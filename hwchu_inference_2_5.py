from ultralytics import YOLO
import cv2

# model_path = 'yolov8m.pt'
# best_model = YOLO("yolov8m_2_5.yaml", task="detect_2_5").load(model_path) # TEST on detect pt model

train_dir = 'train3'
try:
    train_idx = int(train_dir[5:])
except:
    train_idx = 1

best_model = YOLO(f'./yolov8_2_5/{train_dir}/weights/best.pt')

print('- [FOR Inference]')
# results = best_model('./ultralytics/assets/bus.jpg')
results = best_model('../datasets/nuScenes_2_5/images/val/n008-2018-08-30-15-31-50-0400__CAM_BACK__1535657649037558.jpg', device='cpu')

# print('- [results[0].boxes]')
# print(results[0].boxes)

print('- [results[0].boxes.data.shape]')
print(results[0].boxes.data.shape)
print(results[0].boxes.data)

print('- [results[0].dists.data.shape]')
print(results[0].dists.data.shape)
print(results[0].dists.data)

'''결과 파일로 저장'''
with open(f"./hwchu_best_model_inf_result/{train_dir}/result_2_5_inf[{train_idx}].txt", "w") as file:
    file.write('[BOXES]\n')
    file.write(str(results[0].boxes) + '\n\n')
    file.write('[DISTS]\n')
    file.write(str(results[0].dists))

'''Visualize'''
plots = results[0].plot(show=False)
cv2.imwrite(f'./hwchu_best_model_inf_result/{train_dir}/result_2_5_inf[{train_idx}].jpg', plots)

print('<<hwchu_inference_2_5 FINISH!>>')