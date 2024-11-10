from ultralytics import YOLO
import cv2
import time

# model_path = 'yolov8m.pt'
# best_model = YOLO("yolov8m_2_5.yaml", task="detect_2_5").load(model_path) # TEST on detect pt model

train_dir = 'train11'
try:
    train_idx = int(train_dir[5:])
except:
    train_idx = 1

best_model = YOLO(f'./yolov8_2_5/{train_dir}/weights/best.pt')

print('- [FOR Inference]')
# results = best_model('./ultralytics/assets/bus.jpg')

# img_file = 'val/n008-2018-08-30-15-31-50-0400__CAM_BACK__1535657649037558.jpg'
# img_file = 'val/n015-2018-09-26-11-17-24+0800__CAM_FRONT__1537932222912460.jpg' ## OK!
# img_file = 'val/n015-2018-10-02-10-50-40+0800__CAM_FRONT__1538448810112460.jpg' ## OK!!
img_file = 'val/n015-2018-10-02-10-56-37+0800__CAM_FRONT__1538449288862471.jpg' ## OK!!
s = time.time()
results = best_model(f'../datasets/nuScenes_2_5/images/{img_file}', device='cpu')
print(f'time(time): {time.time()-s:.4f} sec')

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
    file.write('[cls] ' + str(results[0].boxes.cls) + '\n')
    file.write('[xywhn] ' + str(results[0].boxes.xywhn) + '\n')
    file.write('[dists] ' + str(results[0].dists.data))

'''Visualize'''
GT_img = cv2.imread(f'../datasets/nuScenes_2_5/GT_imgs/{img_file}')
plots = results[0].plot(show=False)
print(GT_img.shape)
print(plots.shape)

GT_img_resized = cv2.resize(GT_img, (int(GT_img.shape[1] * (plots.shape[0] / GT_img.shape[0])), plots.shape[0]))

GT_plots_combined = cv2.hconcat([GT_img_resized, plots])

# cv2.imwrite(f'./hwchu_best_model_inf_result/{train_dir}/result_2_5_inf[{train_idx}].jpg', plots)
cv2.imwrite(f'./hwchu_best_model_inf_result/{train_dir}/result_2_5_inf[{train_idx}].jpg', GT_plots_combined)

print('<<hwchu_inference_2_5 FINISH!>>')