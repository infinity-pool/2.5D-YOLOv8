from ultralytics import YOLO

m_scale = 's'
baseline_model = YOLO(f'yolov8{m_scale}.pt')
revised_model = model = YOLO(f"yolov8{m_scale}_2_5.yaml", task="detect_2_5").load(f'yolov8{m_scale}.pt')

# print('[TOTAL PARAMETERS]')
# print('--BASELINE--')
# for param in baseline_model.parameters():
#     print(param)
# print('\n--REVISED MODEL--')
# for param in revised_model.parameters():
#     print(param)

# print('\n\n[PARAMS FOR EACH LAYERS]')
# print('--BASELINE--')
# for name, param in baseline_model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # 예시로 일부 값만 출력
# print('\n--REVISED MODEL--')
# for name, param in revised_model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")  # 예시로 일부 값만 출력


baseline_total_params = sum(p.numel() for p in baseline_model.parameters())
print(f"Total parameters[BASELINE]: {baseline_total_params}")

revised_total_params = sum(p.numel() for p in revised_model.parameters())
print(f"Total parameters[REVISED]: {revised_total_params}")