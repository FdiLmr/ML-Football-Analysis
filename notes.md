What I did so far:

- Use Yolov8m for inference, then fine-tuned it on a dataset from roboflow.
- Set up the project environment and installed necessary dependencies
- Created a .gitignore file to exclude unnecessary files and directories
- Prepared a data.yaml file for the YOLO model, specifying class names and dataset paths
- Obtained a pre-trained YOLOv8m model
- Fine-tuned the YOLOv8m model on a custom football players detection dataset from Roboflow
- Saved the best performing model as 'best.pt' in the 'models' directory
- Implemented a basic inference script (yolo_inference.py) to test the model on a video file
- Successfully ran inference on a sample video ('08fd33_4.mp4') and saved the results
- Created utility functions for reading and saving videos (utils/video_utils.py)
- Implemented a main script (main.py) that uses these utility functions to read an input video and save an output video
- Set up the project structure with separate directories for input videos, output videos, and utility functions
- Updated the requirements.txt file to include necessary dependencies (ultralytics, roboflow, ipykernel, cv2)

Next steps:
- Integrate the YOLOv8 model inference into the main script
- Implement object tracking to follow players across frames
- Add visualization of bounding boxes and player IDs on the output video
- Optimize the code for better performance on longer videos
- Implement error handling and logging for better debugging
