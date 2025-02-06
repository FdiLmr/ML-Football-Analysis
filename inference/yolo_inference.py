from ultralytics import YOLO

model = YOLO('../models/best_06_02.pt') 

results = model.predict('But_roulette.mp4', save=True)
print(results[0])
print('====================================================================')
for box in results[0].boxes:
    print(box)