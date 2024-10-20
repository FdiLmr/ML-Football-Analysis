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

- perspective transformation to map the field from a bird's eye view (essentially view from up top)
- field width : 68m | field length : 105 m
- !!! pixels of field are input manually (in self.pixel_vertices in view_transformer.py), find a way to calc them automatically after

- speed and distance estimator : estimate and visualize the speed and distance traveled by players and display it on screen





## Stuff to consider relative to indoor football case :
- No need for camera movement
- Calculate the static field features, probably with masks or filters ? 
- No referee so just ignore that
- Goalkeepers are not fixed players, don't think that will cause a problem but we'll see
- Sometimes not everyone has the same shirt on, will think that through
- Would be nice to draw bird's eye view on an actual small mini-map like in football games
- Get statistics dashboard afterward ? with multiple actual football stats => problem : how to identify players since id keeps changing, how to detect goals, assists, interceptions etc etc
- How to handle the little corner that is just below the camera ?
- How to avoid players detected outside of indoor field since we sometimes see them walking outside ?

### Workflow rough idea

As this project continues to evolve, there are several potential enhancements and functionalities that could be implemented:

1. **Fine-Tuning on Indoor Football Dataset**: 
   - Aim to fine-tune the YOLOv8 model on a dataset of indoor football videos to improve detection accuracy in this specific context. This will involve collecting and annotating a diverse set of indoor football footage to train the model effectively.

2. **Frontend (User Interface)**:
   - **Website or App**: Develop the frontend using web frameworks like React, Angular, or Vue.js for dynamic user interaction.
   - **Upload Interface**: Implement a simple drag-and-drop interface or file input form for users to upload their indoor football footage. Integrate the AWS S3 SDK to allow users to directly upload files to an S3 bucket from the frontend, using AWS S3 Signed URLs for secure handling of uploads.

3. **Backend (Processing Workflow)**:
   - **AWS Lambda**: Set up a Lambda function to trigger upon file upload to S3. When a user uploads footage, this event triggers the Lambda function that starts the processing workflow.
   - **AWS S3**: Store the uploaded video in a dedicated S3 bucket for raw footage.
   - **AWS Step Functions**: Orchestrate the entire processing pipeline using AWS Step Functions, allowing you to chain multiple Lambda functions and other AWS services to handle video processing, YOLOv8 analysis, and data aggregation.

4. **Video Processing Pipeline**:
   - **YOLOv8 Object Detection**: Deploy the YOLOv8 model on an EC2 instance (or use AWS SageMaker for a more managed solution) to process the video file from S3, detecting players, the ball, and other objects.
   - **AWS Batch**: Utilize AWS Batch for distributed computing if the workload is large, scaling the YOLOv8 analysis over multiple instances.
   - **FFmpeg for Video Processing**: Use FFmpeg or a similar tool for video editing tasks like overlaying bounding boxes, drawing player trajectories, or adding annotations to the output video.

5. **Data Analysis and Statistics**:
   - After processing, extract statistics such as the number of passes, ball possession, and player movement heatmaps. Store and compute these stats using AWS Lambda or a Python Flask API that interacts with your analysis module.
   - **AWS DynamoDB**: Store metadata (e.g., game statistics, player data) in a NoSQL database like DynamoDB for quick retrieval and querying.

6. **Post-Processing and Output**:
   - Store the final edited footage, along with stats and analysis, in an S3 bucket. Use another Lambda function or API endpoint to notify the user (via email or in-app notification) that their footage is ready for download or viewing, providing a download link via S3 signed URLs.

7. **User Authentication and Session Management**:
   - Implement authentication using AWS Cognito for secure user login, signup, and session management, with options for social login like Google or Facebook.

8. **Monitoring and Logging**:
   - Set up AWS CloudWatch to monitor your Lambda functions, EC2 instances, and overall system health, configuring alarms to notify you in case of failures. Use AWS X-Ray for tracing and debugging your application.

9. **Cost Optimization and Scalability**:
   - Utilize AWS Lambda and Step Functions for auto-scaling based on demand, ensuring you only pay for what you use. Implement S3 object lifecycle policies to archive old footage or automatically delete unused files after a certain period to save on storage costs. Use AWS CloudFront to deliver the final video quickly and efficiently to users around the world.

### Tech Stack Overview:
- **Frontend**: React.js (or similar), HTML/CSS, AWS S3 SDK for uploads
- **Backend**: AWS Lambda, AWS Step Functions, AWS S3, EC2 (for YOLOv8), AWS Batch (optional), DynamoDB, API Gateway
- **Authentication**: AWS Cognito
- **Processing**: EC2 or SageMaker for running YOLOv8 models, FFmpeg for video manipulation
- **Storage**: AWS S3 (for raw and processed videos), DynamoDB (for game statistics)
- **Notifications**: AWS SNS or SES for user alerts

### Workflow:
1. User uploads video -> S3 triggers Lambda.
2. Lambda triggers Step Functions -> Processing pipeline starts.
3. YOLOv8 analyzes the video on EC2 or SageMaker.
4. Post-processing and statistics calculation occurs.
5. Final video and stats uploaded to S3.
6. User gets notified with a link to the results.

More to come !
