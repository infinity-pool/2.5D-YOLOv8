from ultralytics import YOLO
import cv2

model_path = 'yolov8n.pt'
model = YOLO("yolov8n_2_5.yaml", task="detect_2_5").load(model_path)
device = 'cuda'

results = model.train(epochs=30,
    device=device,
    data="nuScenes_2_5.yaml",
    project='yolov8_2_5',
    # save_dir="/root/data/hwchu/yolov8_results",
    # workers=0
    ) # nuScenes_2_5.yaml training
# results = model.train(epochs=5, device=device, data="coco8.yaml") # coco8.yaml training

# print('[results[0].boxes.data.shape]')
# print(results[0].boxes.data.shape)

saved_best_model = YOLO(f'{results.save_dir}/weights/best.pt')
# saved_best_model = YOLO('/root/data/hwchu/yolov8_results/weights/best.pt')

# inference_results = saved_best_model('./ultralytics/assets/bus.jpg')
inference_results = saved_best_model('../datasets/nuScenes_2_5/images/val/n008-2018-08-30-15-31-50-0400__CAM_BACK__1535657649037558.jpg')

'''결과 파일로 저장'''
with open("result_2_5_train.txt", "w") as file:
    file.write(str(inference_results[0].boxes))

'''Visualize'''
plots = inference_results[0].plot(show=False)
cv2.imwrite('result_2_5_train.jpg', plots)