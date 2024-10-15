from utils import read_video, save_video
from trackers import Tracker
import time
import os
import cv2



def main():
    start_time = time.time()  # Start timing

    # Read Video
    input_video_path = 'input_videos/08fd33_4.mp4'
    video_frames = read_video(input_video_path)

    # Get video size and duration
    video_size = os.path.getsize(input_video_path)  # Get file size in bytes
    video_capture = cv2.VideoCapture(input_video_path)
    video_duration = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / video_capture.get(cv2.CAP_PROP_FPS)  # Duration in seconds
    video_capture.release()  # Release the video capture object

    # Initialize Tracker
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    
    # Save Video
    output_video_path = input_video_path.replace('.mp4', '_output.avi').replace('input_videos', 'output_videos')  # Updated line
    save_video(output_video_frames, output_video_path)  # Updated line
    

    end_time = time.time()  # End timing
    print(f"Time taken to process the video: {end_time - start_time:.2f} seconds")  # Print time taken
    print(f"Video size: {video_size / (1024 * 1024):.2f} MB")  # Print size in MB
    print(f"Video duration: {video_duration:.2f} seconds")  # Print duration in seconds
if __name__ == "__main__":
    main()