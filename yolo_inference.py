from ultralytics import YOLO

model = YOLO('yolov8m') # Chose the v8m model bc medium size, I don't have a supercomputer

results = model.predict('input_videos/08fd33_4.mp4', save=True)
print(results[0])
print('====================================================================')
for box in results[0].boxes:
    print(box)