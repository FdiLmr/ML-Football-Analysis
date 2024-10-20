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

- Implemented a Tracker class in trackers/tracker.py:
  - Initialized with a YOLO model and ByteTrack tracker
  - Added methods for detecting objects in frames and getting object tracks
  - Implemented functionality to save and load tracking results as stubs for faster processing
- Updated main.py to use the Tracker class for object detection and tracking
- Added support for tracking players, referees, and the ball separately
- Created an __init__.py file in the trackers directory to make the Tracker class easily importable

- Enhanced the Tracker class with additional functionality:
  - Added methods to draw annotations on video frames:
    - draw_ellipse: Draws ellipses around detected objects
    - draw_annotations: Applies annotations to all frames in the video
  - Improved object classification by converting goalkeeper detections to player objects
  - Updated main.py to utilize the new Tracker features

- Added ball tracking with triangle => unreliable so far, sometimes the cursor disappears => we'll fix this
- Added track_id number to players
- Better handling of output videos name and folder
- Some issues when applying to indoor dataset, will check later => prob fine tuned worked too well on professional dataset and therefore not working on indoor anymore

- got cropped image of a player => segmentation with kmeans clustering => separate player and background
- implemented that in the main function to assign team colors to players and ellipse
- improved ball tracking with interpolation
- added tracking of player in possession of the ball
- added team ball control percentages
- goalkeeper can be put in wrong team since they have different shirt color => will think of a solution later

- camera movement : detect corners from the field (top & bottom for example) and extract these features => detect how much these features move => no need for indoor football since camera is fixed ?
- read from stubs bc time consuming operations
- problems : memory issue  " File "d:\ML Football Analysis\camera_movement_estimator\camera_movement_estimator.py", line 111, in draw_camera_movement
    overlay = frame.copy()
              ^^^^^^^^^^^^
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 5.93 MiB for an array with shape (1080, 1920, 3) and data type uint8 " 
- solution : added a line to resize the frames to a smaller resolution (960x540) to reduce memory usage, memory error handling and scaled the text to the resized frame
- other solution : free up RAM on my pc, takes about 50 sec to process for an input vid of 19 MB of duration 30 seconds (so far)



More to come !
