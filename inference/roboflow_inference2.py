import cv2
import json
import time

# ------------------
# 1. Load the JSON
# ------------------
with open("video_results.json", "r") as f:
    data = json.load(f)

frame_offsets = data["frame_offset"]  # e.g. [0, 5, 10, 15, ...]
detections = data["indoor_football_detection"]
# ^ This array should have one entry for each element in frame_offsets

assert len(frame_offsets) == len(detections), (
    "Mismatch between frame_offset length and indoor_football_detection length!"
)

# ----------------------------------------------------
# 2. Build a dict: frame_index -> list_of_predictions
# ----------------------------------------------------
frame_to_boxes = {}
for i, frame_idx in enumerate(frame_offsets):
    # Each detection entry might look like:
    # {
    #   "time": 0.2283407,
    #   "image": {"width": 1280, "height": 720},
    #   "predictions": [
    #       {
    #           "x": 975.0,
    #           "y": 188.0,
    #           "width": 34.0,
    #           "height": 94.0,
    #           "confidence": 0.8967,
    #           "class": "person",
    #           "class_id": 1,
    #           ...
    #       },
    #       ...
    #   ]
    # }
    detection_info = detections[i]
    predictions = detection_info.get("predictions", [])
    frame_to_boxes[frame_idx] = predictions

# ----------------------------------
# 3. Open your original video in CV2
# ----------------------------------
input_video_path = "But_roulette.mp4"
output_video_path = "annotated_But_roulette2.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Could not open {input_video_path}")
    exit(1)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# -------------------------------------------------------------
# 4. Loop over frames, draw boxes if this frame has predictions
# -------------------------------------------------------------
frame_idx = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # no more frames

    # Notify every 10 frames in the terminal
    if frame_idx % 10 == 0:
        print(f"Processed {frame_idx} frames so far...")

    # Check if we have bounding boxes for this frame
    if frame_idx in frame_to_boxes:
        predictions = frame_to_boxes[frame_idx]
        
        for pred in predictions:
            x_center = pred["x"]
            y_center = pred["y"]
            w = pred["width"]
            h = pred["height"]
            label = pred["class"]
            conf  = pred["confidence"]

            # Convert center (x,y) + width,height to top-left,bottom-right
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            # Pick the color:
            #   Red for ball, Blue for person, something else for others
            if label == "ball":
                color = (0, 0, 255)  # BGR for Red
            elif label == "person":
                color = (255, 0, 0)  # BGR for Blue
            else:
                color = (0, 255, 0)  # e.g., green for anything else

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label + confidence
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Write the (annotated) frame to output
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

end_time = time.time()
total_time = end_time - start_time

print(f"Done! Annotated video saved to {output_video_path}")
print(f"Processed {frame_idx} frames in {total_time:.2f} seconds.")