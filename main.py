from ultralytics import YOLO
import cv2
import os
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image , ImageDraw , ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

# Settings
MODEL_PATH = "Data/yolo11x-pose.pt"  # Use pose model
VIDEO_PATH = "Data/input.mp4"
CAPTURE_DIR = "Trail Captured"
OUTPUT_VIDEO_PATH = f"{CAPTURE_DIR}/output.mp4"  # Video in CAPTURE_DIR
PERSON_CLASS_ID = 0  # COCO class ID for "person"
RANDOM_SNAPSHOT_PROB = 0.15
MAX_BB_SNAPSHOTS = 2  # Two random BB images
MAX_KP_SNAPSHOTS = 2  # Two random skeleton images
CONF_THRESHOLD = 0.5
KEYPOINT_CONF_THRESHOLD = 0.5  # Confidence threshold for skeleton lines
HEAD_CIRCLE_RADIUS = 20  # Pixels for head circle
TRAIL_THICKNESS = 2  # Thickness of trail lines
TRAIL_MAX_POINTS = 100  # Limit trail length to avoid clutter
TRAIL_TIMEOUT = 3.0  # Seconds before clearing trail
FONT_PATH = "Data/Vazirmatn-Regular.ttf"  # Path to Farsi font
SIDE_PANEL_WIDTH = 176  # Width of the side panel for counter and table
TABLE_ROW_HEIGHT = 60  # Height of each table row (face crop + ID)
FACE_CROP_SIZE = 50  # Size of face crop square

# COCO pose keypoint connections for skeleton (excluding head)
SKELETON = [
    (5 , 6) , (5 , 7) , (7 , 9) , (6 , 8) , (8 , 10) ,  # Arms
    (5 , 11) , (6 , 12) , (11 , 12) ,  # Torso
    (11 , 13) , (12 , 14) , (13 , 15) , (14 , 16)  # Legs
]

# Create capture directory
Path(CAPTURE_DIR).mkdir(parents = True , exist_ok = True)

# Verify font exists
if not os.path.exists(FONT_PATH) :
    raise FileNotFoundError(f"Farsi font {FONT_PATH} not found. Please provide a valid font file.")

# Load YOLO pose model
model = YOLO(MODEL_PATH)

# Initialize video capture
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Failed to open video source: {VIDEO_PATH}")

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
extended_width = frame_width + SIDE_PANEL_WIDTH  # Extended frame width

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for mp4
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH , fourcc , fps , (extended_width , frame_height))
if not out.isOpened() :
    raise ValueError(f"Failed to initialize video writer for {OUTPUT_VIDEO_PATH}")

# Dictionary to store image tracking and trails
person_images = {}

# Function to generate consistent random color per track ID
def get_trail_color(track_id) :
    random.seed(track_id)  # Seed with track_id for consistency
    return (random.randint(0 , 255) , random.randint(0 , 255) , random.randint(0 , 255))

# Load Farsi font for Pillow
font = ImageFont.truetype(FONT_PATH , 20)  # Smaller font for ID and table
counter_font = ImageFont.truetype(FONT_PATH , 24)  # Larger font for counter

# Helper function to draw Farsi text with Pillow
def draw_farsi_text(frame , text , position , color , font , align_right = False):
    # Convert OpenCV frame (BGR) to Pillow image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    
    # Process Farsi text
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    
    # Convert color from BGR to RGB
    color_rgb = (color[2] , color[1] , color[0])
    
    # Calculate text position for right alignment
    if align_right :
        bbox = draw.textbbox((0 , 0) , bidi_text , font = font)
        text_width = bbox[2] - bbox[0]
        x, y = position
        x = x - text_width  # Align right edge to x
        position = (x , y)
    
    # Draw text
    draw.text(position , bidi_text, font = font , fill = color_rgb)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(image) , cv2.COLOR_RGB2BGR)

# Helper function to draw counter and table
def draw_counter_and_table(frame , person_count , person_images) :
    # Convert to Pillow image
    image = Image.fromarray(cv2.cvtColor(frame , cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    
    # Draw counter
    counter_text = f"تعداد افراد: {person_count}"
    reshaped_counter = arabic_reshaper.reshape(counter_text)
    bidi_counter = get_display(reshaped_counter)
    draw.text((frame_width + 10, 20) , bidi_counter, font = counter_font , fill = (0, 0, 0))
    
    # Draw table
    y_offset = 60  # Start table below counter
    for track_id, data in person_images.items() :
        if y_offset + TABLE_ROW_HEIGHT > frame_height :
            print("Warning: Table overflowed frame height.")
            break
            
        # Draw ID
        id_text = f"آیدی: {track_id}"
        reshaped_id = arabic_reshaper.reshape(id_text)
        bidi_id = get_display(reshaped_id)
        draw.text((frame_width + 60 , y_offset + 15) , bidi_id, font = font , fill = (0 , 0 , 0))
        
        # Draw face crop if available
        if data["face_crop"] is not None:
            face_crop = data["face_crop"]
            # Resize face crop to fit table
            face_crop_resized = cv2.resize(face_crop , (FACE_CROP_SIZE , FACE_CROP_SIZE))
            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop_resized , cv2.COLOR_BGR2RGB))
            image.paste(face_crop_pil , (frame_width + 5 , y_offset))
        
        y_offset += TABLE_ROW_HEIGHT
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(image) , cv2.COLOR_RGB2BGR)

try:
    while True:
        ret, frame = cap.read()
        if not ret :
            print("End of video or capture failed.")
            break

        current_time = time.time()

        # Create extended frame with side panel
        extended_frame = np.zeros((frame_height , extended_width, 3) , dtype = np.uint8)
        extended_frame[ : , : frame_width] = frame  # Copy original frame
        extended_frame[ : , frame_width : ] = (255 , 255 , 255)  # White side panel

        # Keep a clean copy of the frame for face cropping
        clean_frame = frame.copy()

        # Run detection and tracking with pose estimation
        results = model.track(frame , persist = True , stream = False , verbose = False , show = False , conf = CONF_THRESHOLD)[0]

        # Initialize detected IDs and counter
        detected_ids = set()
        person_count = 0

        # Skip if no boxes or no tracking IDs
        if not results.boxes or results.boxes.id is None :
            # Check for stale trails and table entries
            for track_id in list(person_images.keys()) :
                if current_time - person_images[track_id]["last_seen"] > TRAIL_TIMEOUT :
                    person_images[track_id]["trail"] = []  # Clear trail
                    person_images.pop(track_id)  # Remove from table
            # Draw counter and empty table
            extended_frame = draw_counter_and_table(extended_frame, person_count , {})
            cv2.imshow("Tracking" , extended_frame)
            out.write(extended_frame)  # Write empty frame to video
            if cv2.waitKey(1) & 0xFF == ord("q") :
                break
            continue

        boxes = results.boxes
        keypoints = results.keypoints

        # Process detections
        for box, track_id_tensor , cls_id , kpts in zip(boxes.xyxy , boxes.id , boxes.cls , keypoints) :
            # Only process "person" class
            if int(cls_id) != PERSON_CLASS_ID:
                continue

            x1 , y1 , x2 , y2 = map(int , box)
            track_id = int(track_id_tensor)
            detected_ids.add(track_id)
            person_count += 1

            # Validate crop coordinates
            h, w = frame.shape[ : 2]
            x1, y1 = max(0 , x1) , max(0 , y1)
            x2, y2 = min(w , x2) , min(h , y2)
            if x2 <= x1 or y2 <= y1 :
                continue  # Skip invalid boxes

            # Initialize tracking state
            if track_id not in person_images :
                person_images[track_id] = {
                    "entry": False ,
                    "bb_snapshots": [] ,
                    "kp_snapshots": [] ,
                    "trail": [] ,  # Store (x, y) points for path
                    "trail_color": get_trail_color(track_id) ,  # Unique color
                    "last_seen": current_time ,
                    "face_crop": None  # Store face crop for table
                }

            # Update last seen time
            person_images[track_id]["last_seen"] = current_time

            # Calculate trail point (middle of bottom side of bounding box)
            trail_x = (x1 + x2) // 2
            trail_y = y2
            person_images[track_id]["trail"].append((trail_x , trail_y))

            # Limit trail length
            if len(person_images[track_id]["trail"]) > TRAIL_MAX_POINTS:
                person_images[track_id]["trail"].pop(0)

            # Create frame copies for saving
            bb_frame = frame.copy()  # For bounding box images
            kp_frame = frame.copy()  # For skeleton images

            # Get color for this track ID
            track_color = person_images[track_id]["trail_color"]

            # Draw bounding box on bb_frame and frame
            cv2.rectangle(bb_frame , (x1 , y1) , (x2 , y2) , track_color , 2)
            cv2.rectangle(frame , (x1 , y1) , (x2 , y2) , track_color , 2)

            # Draw skeleton lines and head circle on kp_frame and frame
            if kpts is not None and kpts.data.shape[0] > 0 :
                kpts_data = kpts.data[0].cpu().numpy()  # Shape: [17, 3] (x, y, conf)
                if kpts_data[0][2] > KEYPOINT_CONF_THRESHOLD:
                    nose_x, nose_y = int(kpts_data[0][0]) , int(kpts_data[0][1])
                    cv2.circle(kp_frame , (nose_x , nose_y) , HEAD_CIRCLE_RADIUS , track_color , 2)
                    cv2.circle(frame , (nose_x , nose_y) , HEAD_CIRCLE_RADIUS , track_color , 2)
                for start_idx , end_idx in SKELETON :
                    if (kpts_data[start_idx][2] > KEYPOINT_CONF_THRESHOLD and 
                        kpts_data[end_idx][2] > KEYPOINT_CONF_THRESHOLD) :
                        start = (int(kpts_data[start_idx][0]) , int(kpts_data[start_idx][1]))
                        end = (int(kpts_data[end_idx][0]) , int(kpts_data[end_idx][1]))
                        cv2.line(kp_frame , start , end , track_color , 2)
                        cv2.line(frame , start , end , track_color , 2)

            # Draw trail on frame
            trail = person_images[track_id]["trail"]
            trail_color = person_images[track_id]["trail_color"]
            for i in range(1 , len(trail)) :
                cv2.line(frame , trail[i-1] , trail[i] , trail_color , TRAIL_THICKNESS)

            # Save entry image (with bounding box)
            if not person_images[track_id]["entry"] :
                entry_path = f"{CAPTURE_DIR}/person_{track_id}_entry.jpg"
                cv2.imwrite(entry_path , bb_frame[y1 : y2 , x1 : x2])
                person_images[track_id]["entry"] = True
                print(f"Saved entry for ID {track_id}")

            # Crop face from clean frame using nose keypoint (update every frame)
            if kpts is not None and kpts.data.shape[0] > 0 :
                kpts_data = kpts.data[0].cpu().numpy()
                if kpts_data[0][2] > KEYPOINT_CONF_THRESHOLD :
                    nose_x , nose_y = int(kpts_data[0][0]) , int(kpts_data[0][1])
                    face_x1 = max(0 , nose_x - FACE_CROP_SIZE // 2)
                    face_y1 = max(0 , nose_y - FACE_CROP_SIZE // 2)
                    face_x2 = min(w , face_x1 + FACE_CROP_SIZE)
                    face_y2 = min(h , face_y1 + FACE_CROP_SIZE)
                    if face_x2 > face_x1 and face_y2 > face_y1 :
                        face_crop = clean_frame[face_y1 : face_y2 , face_x1 : face_x2].copy()
                        person_images[track_id]["face_crop"] = face_crop

            # Draw Farsi ID text at top-right of bounding box
            frame = draw_farsi_text(frame , f"آیدی {track_id}" , (x2, y1 - 30) , track_color , font , align_right = True)
            bb_frame = draw_farsi_text(bb_frame , f"آیدی {track_id}" , (x2 , y1 - 30) , track_color , font , align_right = True)

            # Save random snapshots
            if (len(person_images[track_id]["bb_snapshots"]) < MAX_BB_SNAPSHOTS or 
                len(person_images[track_id]["kp_snapshots"]) < MAX_KP_SNAPSHOTS) :
                if random.random() < RANDOM_SNAPSHOT_PROB:
                    if (len(person_images[track_id]["bb_snapshots"]) < MAX_BB_SNAPSHOTS and
                        (len(person_images[track_id]["kp_snapshots"]) >= MAX_KP_SNAPSHOTS or
                         random.choice([True , False]))) :
                        snap_id = len(person_images[track_id]["bb_snapshots"]) + 1
                        path = f"{CAPTURE_DIR}/person_{track_id}_bb_random{snap_id}.jpg"
                        cv2.imwrite(path, bb_frame[y1:y2, x1 : x2])
                        person_images[track_id]["bb_snapshots"].append(path)
                        print(f"Saved BB snapshot {snap_id} for ID {track_id}")
                    elif len(person_images[track_id]["kp_snapshots"]) < MAX_KP_SNAPSHOTS :
                        snap_id = len(person_images[track_id]["kp_snapshots"]) + 1
                        path = f"{CAPTURE_DIR}/person_{track_id}_kp_random{snap_id}.jpg"
                        cv2.imwrite(path, kp_frame[y1 : y2 , x1 : x2])
                        person_images[track_id]["kp_snapshots"].append(path)
                        print(f"Saved KP snapshot {snap_id} for ID {track_id}")

        # Check for stale trails and table entries
        for track_id in list(person_images.keys()) :
            if track_id not in detected_ids and current_time - person_images[track_id]["last_seen"] > TRAIL_TIMEOUT :
                person_images[track_id]["trail"] = []  # Clear trail
                person_images.pop(track_id)  # Remove from table

        # Copy modified frame to extended frame
        extended_frame[ : , : frame_width] = frame

        # Draw counter and table on side panel
        extended_frame = draw_counter_and_table(extended_frame , person_count , person_images)

        cv2.imshow("Tracking" , extended_frame)
        out.write(extended_frame)  # Write frame to output video
        if cv2.waitKey(1) & 0xFF == ord("q") :
            break

finally :
    # Cleanup
    cap.release()
    out.release()  # Close video writer
    cv2.destroyAllWindows()