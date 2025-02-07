from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
        
        
    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1)
            detections += detections_batch 
        return detections
        
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        detections = self.detect_frames(frames)
        
        tracks = {
            "persons":[],       
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names                             #{0:person, 1:goalkeeper, ...}
            cls_names_inv = {v:k for k,v in cls_names.items()}     #{person:0, goalkeeper:1, ...}
            print(cls_names)
            
            #Convert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            #Convert Goalkeeper to person object
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["person"]
                    
            #Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)
            
            tracks["persons"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]
                
                if cls_id == cls_names_inv['person']:
                    tracks["persons"][frame_num][track_id] = {"bbox":bbox}
                
                #if cls_id == cls_names_inv['referee']:
                    #tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                    
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}
        
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
            
        return tracks


    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Draw an improved ellipse
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA  # Anti-aliased lines for smoothness
        )

        # Define rectangle dimensions
        rectangle_width = 24
        rectangle_height = 14
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y2 + 15 - rectangle_height // 2
        y2_rect = y2 + 15 + rectangle_height // 2

        if track_id is not None:
            # Draw a rectangle with better contrast
            cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color, cv2.FILLED)

            # Improve text positioning and visibility
            text = f"{track_id}"
            font_scale = 0.4  # Adjusted font size
            font_thickness = 1
            font = cv2.FONT_HERSHEY_DUPLEX
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = x_center - text_size[0] // 2  # Center the text horizontally
            text_y = y1_rect + (rectangle_height // 2) + (text_size[1] // 2)  # Adjust vertically

            # Add a text outline for better contrast
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)  # Outline
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)  # Main text

        return frame

    
    def draw_triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
        team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        
        return frame
        
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        num_tracked_frames = len(tracks["persons"])  # Get the length of tracking data

        for frame_num, frame in enumerate(video_frames):
            if frame_num >= num_tracked_frames:
                print(f"Skipping frame {frame_num} as there is no tracking data available.")
                break  # Exit the loop when tracking data ends

            frame = frame.copy()

            person_dict = tracks["persons"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            #referee_dict = tracks["referees"][frame_num]

            # Draw persons
            for track_id, person in person_dict.items():
                color = person.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, person["bbox"], color, track_id)

                if person.get('has_ball', False):
                    frame = self.draw_triangle(frame, person["bbox"], (0, 0, 255))

            # Draw Ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw Team Ball Control
            if frame_num < len(team_ball_control):  # Ensure team_ball_control is available
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
