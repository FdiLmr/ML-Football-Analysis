from utils import read_video, save_video
from trackers import Tracker
import time
import os
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator



def main():
    start_time = time.time()  # Start timing

    # Read Video
    input_video_path = 'input_videos/But de Team A.mp4'
    video_frames = read_video(input_video_path)

    # Get video size and duration
    video_size = os.path.getsize(input_video_path)  # Get file size in bytes
    video_capture = cv2.VideoCapture(input_video_path)
    video_duration = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / video_capture.get(cv2.CAP_PROP_FPS)  # Duration in seconds
    video_capture.release()  # Release the video capture object

    # Extract video name without extension
    video_name = os.path.splitext(os.path.basename(input_video_path))[0]

    # Define stub file paths dynamically
    track_stub_path = f'stubs/{video_name}_track_stubs.pkl'
    camera_movement_stub_path = f'stubs/{video_name}_camera_movement_stub.pkl'

    # Initialize Tracker
    tracker = Tracker('models/best_06_02.pt')

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path=track_stub_path)

    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path=camera_movement_stub_path
    )
    
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)
    
    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['persons'][0])
    
    for frame_num, player_track in enumerate(tracks['persons']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['persons'][frame_num][player_id]['team'] = team 
            tracks['persons'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['persons']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['persons'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['persons'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    ## Draw Camera movement
    #output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)
    
    # Save Video
    output_video_path = input_video_path.replace('.mp4', '_indoor_output.avi').replace('input_videos', 'output_videos')  # Updated line
    save_video(output_video_frames, output_video_path)  # Updated line
    

    end_time = time.time()  # End timing
    print(f"Time taken to process the video: {end_time - start_time:.2f} seconds")  # Print time taken
    print(f"Video size: {video_size / (1024 * 1024):.2f} MB")  # Print size in MB
    print(f"Video duration: {video_duration:.2f} seconds")  # Print duration in seconds
if __name__ == "__main__":
    main()