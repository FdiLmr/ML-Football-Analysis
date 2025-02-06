import cv2
import os
from dotenv import load_dotenv
import roboflow
import time

load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")

# Initialize Roboflow and model
rf = roboflow.Roboflow(api_key=api_key)
project = rf.workspace().project("indoor_football_detection")
model = project.version("5").model
model.confidence = 50
model.overlap = 25

# Path to your input video
input_video_path = "But_roulette.mp4"
# Path to your output (annotated) video
output_video_path = "annotated_But_roulette.mp4"

# Open the input video using OpenCV
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Could not open {input_video_path}")
    exit(1)

# Get some video details
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Set up the VideoWriter for the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or "avc1", etc.
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # No more frames in the video

    # 1. Save the frame to a temporary file (JPEG or PNG)
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame)

    # 2. Inference on this frame
    predictions = model.predict(temp_frame_path).json()

    # 3. Draw bounding boxes from the predictions
    #    Typically, predictions["predictions"] is a list of detection dicts
    for pred in predictions.get("predictions", []):
        # Coordinates
        x     = pred["x"]
        y     = pred["y"]
        w     = pred["width"]
        h     = pred["height"]
        label = pred["class"]
        conf  = pred["confidence"]

        # Convert center x,y + width,height to top-left and bottom-right
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        # Draw rectangle
        color = (0, 255, 0)  # green
        thickness = 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw label text above the box
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 4. Write the annotated frame to the output video
    out.write(frame)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
out.release()

end_time = time.time()
print(f"Done! Processed {frame_count} frames in {end_time - start_time:.2f} seconds.")
print(f"Annotated video saved to: {output_video_path}")


# Done! Processed 800 frames in 328.31 seconds. (32 sec video with 25 fps)