from ultralytics import YOLO
import cv2

test_model_scales = ['n', 's', 'm', 'l']
epochs = 40

for m_scale in test_model_scales:
    for epoch in epochs:
        model_path = f'yolov8{m_scale}.pt'
        model = YOLO(f"yolov8{m_scale}_2_5.yaml", task="detect_2_5").load(model_path)
        device = 'cuda'

        results = model.train(epochs=epoch,
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

