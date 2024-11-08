from ultralytics import YOLO
import cv2

def freeze_layer(trainer):
    model = trainer.model
    num_freeze = 22
    print(f"Freezing {num_freeze} layers")
    freeze = [f'model.{x}.' for x in range(num_freeze)]  # layers to freeze 
    for k, v in model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            print(f'freezing {k}') 
            v.requires_grad = False 
    print(f"{num_freeze} layers are freezed.")

# test_model_scales = ['n', 's', 'm', 'l']
m_scale = 'm'
epochs = 40

model_path = f'yolov8{m_scale}.pt'
model = YOLO(f"yolov8{m_scale}_2_5.yaml", task="detect_2_5").load(model_path)
device = 'cuda'

## model freeze test
model.add_callback("on_train_start", freeze_layer)

results = model.train(epochs=epochs,
    device=device,
    data="nuScenes_2_5.yaml",
    project='yolov8_2_5',
    # save_dir="/root/data/hwchu/yolov8_results",
    # workers=0
    )

saved_best_model = YOLO(f'{results.save_dir}/weights/best.pt')

# inference_results = saved_best_model('./ultralytics/assets/bus.jpg')
inference_results = saved_best_model('../datasets/nuScenes_2_5/images/val/n008-2018-08-30-15-31-50-0400__CAM_BACK__1535657649037558.jpg')

'''결과 파일로 저장'''
with open(f"{results.save_dir}/result_2_5_train.txt", "w") as file:
    file.write(str(inference_results[0].boxes))
    file.write('[BOXES]\n')
    file.write(str(inference_results[0].boxes) + '\n\n')
    file.write('[DISTS]\n')
    file.write(str(inference_results[0].dists))

'''Visualize'''
plots = inference_results[0].plot(show=False)
cv2.imwrite(f'{results.save_dir}/result_2_5_train.jpg', plots)

