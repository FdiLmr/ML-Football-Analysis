from utils import read_video, save_video
from trackers import Tracker
import time
import os
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner




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
    
    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
    
    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control= np.array(team_ball_control)
    
    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    
    # Save Video
    output_video_path = input_video_path.replace('.mp4', '_output.avi').replace('input_videos', 'output_videos')  # Updated line
    save_video(output_video_frames, output_video_path)  # Updated line
    

    end_time = time.time()  # End timing
    print(f"Time taken to process the video: {end_time - start_time:.2f} seconds")  # Print time taken
    print(f"Video size: {video_size / (1024 * 1024):.2f} MB")  # Print size in MB
    print(f"Video duration: {video_duration:.2f} seconds")  # Print duration in seconds
if __name__ == "__main__":
    main()