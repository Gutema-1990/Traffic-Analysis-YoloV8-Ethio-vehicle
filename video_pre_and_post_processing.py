import cv2
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import time, math
from ultralytics import YOLO
import supervision as sv

# Global variables for line drawing
drawing = False  # True if the mouse is pressed
start_points = []  # List to store start points of lines
end_points = []    # List to store end points of lines
current_line = 0   # Index of the current line being drawnsplit
frame = None
model = YOLO("latest_model_openvino_model", task='detect')  # Load the YOLO model

vehicle_type_names ={0:'bus', 
                      1:'car', 
                      2:'minibus', 
                      3:'motorcycle', 
                      4:'pickup', 
                      5:'truck'}
vehicle_type_counts_towards = defaultdict(int)  # Count of vehicles moving towards the line
vehicle_type_counts_away = defaultdict(int)     # Count of vehicles moving away from the line
crossed_vehicle_ids = set()  # Track vehicles that have already crossed the line
vehicle_previous_positions = {}  # Store previous positions of tracked vehicles


def save_prediction(dict_tobe_saved, save_path, save_prediction_in, direction):
    print("Path:", save_path)
    print("Data format:", save_prediction_in)

    # Create DataFrame with vehicle types as index and counts as values
    df = pd.DataFrame.from_dict(dict_tobe_saved, orient='index', columns=['Count']).reset_index()
    df.rename(columns={'index': 'Vehicle Type'}, inplace=True)
   
    def get_unique_filename(filename, extension):
        base, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = f"{base}{ext}"
        while os.path.exists(unique_filename):
            unique_filename = f"{base}_{counter}{ext}"
            counter += 1
        return unique_filename
    
    if direction == "away":
        if save_prediction_in == 'csv':
            csv_filename = os.path.join(save_path, 'data_away.csv')
            unique_csv_filename = get_unique_filename(csv_filename, '.csv')
            df.to_csv(unique_csv_filename, index=False)
            print("Saved In CSV file Successfully:", unique_csv_filename)

        if save_prediction_in == 'excel':
            excel_filename = os.path.join(save_path, 'data_away.xlsx')
            unique_excel_filename = get_unique_filename(excel_filename, '.xlsx')
            df.to_excel(unique_excel_filename, index=False)
    
    if direction == "toward":
        if save_prediction_in == 'csv':
            csv_filename = os.path.join(save_path, 'data_toward.csv')
            unique_csv_filename = get_unique_filename(csv_filename, '.csv')
            df.to_csv(unique_csv_filename, index=False)
            print("Saved In CSV file Successfully:", unique_csv_filename)

        if save_prediction_in == 'excel':
            excel_filename = os.path.join(save_path, 'data_toward.xlsx')
            unique_excel_filename = get_unique_filename(excel_filename, '.xlsx')
            df.to_excel(unique_excel_filename, index=False)
    if direction == '2_line':
        if save_prediction_in == 'csv':
            csv_filename = os.path.join(save_path, 'data.csv')
            unique_csv_filename = get_unique_filename(csv_filename, '.csv')
            df.to_csv(unique_csv_filename, index=False)
            print("Saved In CSV file Successfully:", unique_csv_filename)

        if save_prediction_in == 'excel':
            excel_filename = os.path.join(save_path, 'data.xlsx')
            unique_excel_filename = get_unique_filename(excel_filename, '.xlsx')
            df.to_excel(unique_excel_filename, index=False)     

# Function to handle mouse events for line drawing
def draw_line(event, x, y, flags, param):
    global drawing, start_points, end_points, current_line, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_points[current_line] = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_points[current_line] = (x, y)
            temp_frame = frame.copy()
            for i in range(current_line + 1):
                cv2.line(temp_frame, start_points[i], end_points[i], (0, 255, 0), 2)
            cv2.imshow('Frame', temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_points[current_line] = (x, y)
        cv2.line(frame, start_points[current_line], end_points[current_line], (0, 255, 0), 2)
        current_line += 1  # Move to the next line
        cv2.imshow('Frame', frame)

# Function to check if two line segments intersect
def do_lines_intersect(p1, p2, q1, q2):
    """ Check if line segment p1-p2 intersects with q1-q2. """
    def orientation(a, b, c):
        """ Helper function to find the orientation of an ordered triplet (a, b, c).
            0 --> a, b and c are collinear
            1 --> Clockwise
            2 --> Counterclockwise """
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0: return 0
        return 1 if val > 0 else 2

    def on_segment(a, b, c):
        """ Check if point b is on segment a-c. """
        if min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1]):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if o1 != o2 and o3 != o4:
        return True

    # Check for special cases when collinear
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True

    return False

def process_video(
                    input_video_path, draw_line_option, sav_prediction, save_prediction_in, save_video,
                    video_speed, pred_save_dir, output_video_path, conf_level
                ):
    global frame, start_points, end_points, current_line, crossed_vehicle_ids, vehicle_previous_positions

    # Initialize start and end points based on the number of lines
    start_points = [(-1, -1)] * draw_line_option
    end_points = [(-1, -1)] * draw_line_option
    current_line = 0

    print("The Function is already called")
    if sav_prediction and pred_save_dir:
        if not os.path.exists(pred_save_dir):
            os.makedirs(pred_save_dir)
    elif sav_prediction:
        print("Please provide a valid Prediction Save Directory")
        return 0

    if save_video and output_video_path:
        output_video_dir = os.path.dirname(output_video_path)
        if not os.path.exists(output_video_dir):
            os.makedirs(output_video_dir)
    elif save_video:
        print("Please provide a valid Output Video Path")
        return 0

    cap = cv2.VideoCapture(input_video_path)
    assert cap.isOpened(), "Error reading video file"

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    # total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    frame_width = 1500
    frame_height = 1200
    # width, height = 640, 480

    # Define the codec and create VideoWriter object    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    def get_unique_filename(filename):
        base, ext = os.path.splitext(filename)
        counter = 1
        unique_filename = f"{base}{ext}"
        while os.path.exists(unique_filename):
            unique_filename = f"{base}_{counter}{ext}"
            counter += 1
        return unique_filename
    
    if save_video:
        output_video_path = os.path.join(output_video_path, 'output_video.avi')
        unique_output_video_path = get_unique_filename(output_video_path)
        out = cv2.VideoWriter(unique_output_video_path, fourcc, fps, (frame_width, frame_height))
    else:
        out = None

    classes_to_count = [0, 1, 2, 3, 4, 5]  # Counting bicycle, car, motorcycle, bus, truck, and pedestrian

    # Read the first frame to draw the line(s)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the first frame from the video")
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        return
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Create a window and set the mouse callback function to draw the line(s)
    cv2.namedWindow('Frame')
    cv2.setMouseCallback('Frame', draw_line)

    # Display the first frame and wait for the lines to be drawn
    while True:
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if the desired number of lines are drawn
        if current_line >= draw_line_option:
            break


        if key == ord('q'):
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            return
    current_frame = 0
    if draw_line_option == 1:

        processed_ids = set()
        crossed_vehicle_ids = set()
        vehicle_type_counts_towards = defaultdict(int)
        vehicle_type_counts_away = defaultdict(int)
        vehicle_previous_positions = {}

        # Process video frames
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            current_frame += 1

  # Calculate elapsed time in seconds based on current frame and fps
            frame = cv2.resize(frame, (frame_width, frame_height))
            elapsed_time_seconds = (current_frame / fps) * video_speed

            for _ in range(int(video_speed) - 1):
                cap.grab()
            
            # elapsed_time_seconds = current_frame / fps
            # if tracks[0].boxes.id is not None:

            results = model.track(frame, persist=True, show=False, classes=classes_to_count, conf=conf_level)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist() 

                # If track_ids is None, skip the current frame

            
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                detections = sv.Detections.from_ultralytics(results[0])

                for vehicle_type, vehicle_id, box in zip(detections.class_id, track_ids, boxes):
                    if vehicle_type in classes_to_count and vehicle_id not in crossed_vehicle_ids:
                        center_x, center_y = box[0].item(), box[1].item()
                        current_center = (center_x, center_y)

                        # Get the corners of the bounding box
                        w, h = box[2].item(), box[3].item()
                        top_left = (center_x - w / 2, center_y - h / 2)
                        top_right = (center_x + w / 2, center_y - h / 2)
                        bottom_left = (center_x - w / 2, center_y + h / 2)
                        bottom_right = (center_x + w / 2, center_y + h / 2)

                        # Edges of the bounding box
                        box_edges = [
                            (top_left, top_right),
                            (top_right, bottom_right),
                            (bottom_right, bottom_left),
                            (bottom_left, top_left)
                        ]

                        # Check if any edge intersects with the line
                        intersecting = False
                        for edge in box_edges:
                            if do_lines_intersect(start_points[0], end_points[0], edge[0], edge[1]):
                                intersecting = True
                                break

                        if intersecting:
                            if vehicle_id not in vehicle_previous_positions:
                                vehicle_previous_positions[vehicle_id] = current_center
                            else:
                                prev_position = vehicle_previous_positions[vehicle_id]

                                # Vector representing the line
                                line_vec = np.array([end_points[0][0] - start_points[0][0], end_points[0][1] - start_points[0][1]])
                                # Vector representing the movement of the vehicle
                                movement_vec = np.array([current_center[0] - prev_position[0], current_center[1] - prev_position[1]])

                                # Determine if the vehicle is moving towards or away from the line
                                if np.dot(line_vec, movement_vec) > 0:
                                    vehicle_type_counts_towards[vehicle_type_names[vehicle_type]] += 1
                                else:
                                    vehicle_type_counts_away[vehicle_type_names[vehicle_type]] += 1

                                crossed_vehicle_ids.add(vehicle_id)  # Mark the vehicle as processed
                                vehicle_previous_positions[vehicle_id] = current_center

            
                # Overlay vehicle counts on the frame
                count_text_towards = "Vehicle Counts (Towards):\n"
                for vehicle_type, count in vehicle_type_counts_towards.items():
                    count_text_towards += f"{vehicle_type}: {count}\n"

                count_text_away = "Vehicle Counts (Away):\n"
                for vehicle_type, count in vehicle_type_counts_away.items():
                    count_text_away += f"{vehicle_type}: {count}\n"

                # Set the position and font for the text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (0, 255, 0)
                line_type = 2
                x, y = 20, 40

                # Draw the line on the frame
                red_color = (0, 0, 255)
                cv2.line(annotated_frame, start_points[0], end_points[0], red_color, 2)
                minutes = int(elapsed_time_seconds // 60)
                seconds_1 = int(elapsed_time_seconds % 60)
                text =  str(minutes) + ":" + str(seconds_1) + "min"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (255, 255, 255)  # White color
                thickness = 3

            # Calculate bottom left corner coordinates (adjust slightly if needed)
                text_x = 10  
                text_y = frame_height - 10  

                # Put text on the frame
                cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, red_color, thickness)



                # Draw the count text on the frame
                for i, line in enumerate(count_text_towards.split('\n')):
                    y_pos = y + (i * 20)
                    cv2.putText(annotated_frame, line, (x, y_pos), font, font_scale, red_color, line_type)

                for i, line in enumerate(count_text_away.split('\n')):
                    y_pos = y + 20 * (len(count_text_towards.split('\n')) + i)
                    cv2.putText(annotated_frame, line, (x, y_pos), font, font_scale, red_color, line_type)

                if save_video and out is not None:
                    out.write(annotated_frame)

                cv2.imshow('Frame', annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break


        # Save the counts if needed
        vehicle_type_counts_towards_1 = {}
        for vehicle, count in vehicle_type_counts_towards.items():
            vehicle_type_counts_towards_1[vehicle] = count

        if sav_prediction:
            save_path = pred_save_dir if pred_save_dir else os.getcwd()
            save_prediction(vehicle_type_counts_towards_1, save_path, save_prediction_in, "toward")
        
        vehicle_type_counts_away_1 = {}
        for vehicle, count in vehicle_type_counts_away.items():
            vehicle_type_counts_away_1[vehicle] = count

        if sav_prediction:
            save_path = pred_save_dir if pred_save_dir else os.getcwd()
            save_prediction(vehicle_type_counts_away_1, save_path, save_prediction_in, "away")

        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
    if draw_line_option == 2:
        current_frame = 0
        
        crossed_vehicle_ids_line1 = set()
        crossed_vehicle_ids_both_lines = set()
        vehicle_type_counts = {vehicle_type: 0 for vehicle_type in vehicle_type_names.values()}

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Video frame is empty or video processing has been successfully completed.")
                break
            current_frame += 1
            elapsed_time_seconds = current_frame / fps

            for _ in range(int(video_speed) - 1):
                cap.grab()

            # Perform object tracking

            results = model.track(frame, persist=True, show=False, classes=classes_to_count)
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
        
                
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                detections = sv.Detections.from_ultralytics(results[0])

                for vehicle_type, vehicle_id, box in zip(detections.class_id, track_ids, boxes):
                    if vehicle_type in classes_to_count:
                        center_x, center_y = box[0].item(), box[1].item()
                        current_center = (center_x, center_y)

                        # Get the corners of the bounding box
                        w, h = box[2].item(), box[3].item()
                        top_left = (center_x - w / 2, center_y - h / 2)
                        top_right = (center_x + w / 2, center_y - h / 2)
                        bottom_left = (center_x - w / 2, center_y + h / 2)
                        bottom_right = (center_x + w / 2, center_y + h / 2)

                        # Edges of the bounding box
                        box_edges = [
                            (top_left, top_right),
                            (top_right, bottom_right),
                            (bottom_right, bottom_left),
                            (bottom_left, top_left)
                        ]

                        # Check if any edge intersects with the first line
                        intersecting_line1 = False
                        for edge in box_edges:
                            if do_lines_intersect(start_points[0], end_points[0], edge[0], edge[1]):
                                intersecting_line1 = True
                                break

                        # Check if any edge intersects with the second line
                        intersecting_line2 = False
                        for edge in box_edges:
                            if do_lines_intersect(start_points[1], end_points[1], edge[0], edge[1]):
                                intersecting_line2 = True
                                break

                        if intersecting_line1 and vehicle_id not in crossed_vehicle_ids_line1:
                            crossed_vehicle_ids_line1.add(vehicle_id)

                        if intersecting_line2 and vehicle_id in crossed_vehicle_ids_line1 and vehicle_id not in crossed_vehicle_ids_both_lines:
                            crossed_vehicle_ids_both_lines.add(vehicle_id)
                            vehicle_type_counts[vehicle_type_names[vehicle_type]] += 1

                # Overlay vehicle counts on the frame
                count_text = "Vehicle Counts:\n"
                for vehicle_type, count in vehicle_type_counts.items():
                    count_text += f"{vehicle_type}: {count}\n"

                # Set the position and font for the text overlay
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_color = (0, 255, 0)
                line_type = 2
                x, y = 20, 40

                # Draw the lines on the frame
                red_color = (0, 0, 255)
                cv2.line(annotated_frame, start_points[0], end_points[0], red_color, 2)
                cv2.line(annotated_frame, start_points[1], end_points[1], red_color, 2)
                minutes = int(elapsed_time_seconds // 60)
                seconds_1 = int(elapsed_time_seconds % 60)
                text =  str(minutes) + ":" + str(seconds_1) + "min"
    # Assuming you have the minutes calculated
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                color = (255, 255, 255)  # White color
                thickness = 3

                # Calculate bottom left corner coordinates (adjust slightly if needed)
                text_x = 10  
                text_y = frame_height - 10  

                # Put text on the frame
                cv2.putText(annotated_frame, text, (text_x, text_y), font, font_scale, red_color, thickness)

                # Draw the count text on the frame
                for i, line in enumerate(count_text.split('\n')):
                    y_pos = y + (i * 20)
                    cv2.putText(annotated_frame, line, (x, y_pos), font, font_scale, red_color, line_type)

                if save_video and out is not None:
                    out.write(annotated_frame)

                cv2.imshow('Frame', annotated_frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

        if sav_prediction:
            print("CXXXXXX-----------", vehicle_type_counts)
            save_path = pred_save_dir if pred_save_dir else os.getcwd()
            save_prediction(vehicle_type_counts, save_path, save_prediction_in, "2_line")

        cap.release()
        cv2.destroyAllWindows()