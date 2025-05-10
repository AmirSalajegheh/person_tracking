# Person Tracking with YOLOv11 Pose Estimation

This project implements a person tracking system using the YOLOv11 pose estimation model. It processes a video input, tracks individuals, draws bounding boxes, skeletons, and movement trails, and displays a side panel with a person counter and a table of tracked IDs with face crops. The system also saves entry images and random snapshots of bounding boxes and skeletons for each tracked person.

A demo video showcasing the tracking functionality is included in the repository as `demo.mp4`.

## Features

- **Person Detection and Tracking**: Uses YOLOv11 pose model to detect and track people in a video.
- **Bounding Boxes and Skeletons**: Draws bounding boxes and skeletal keypoints (excluding head) for each person.
- **Movement Trails**: Visualizes movement paths with unique colors per person, limited to 100 points or 3 seconds.
- **Side Panel**: Displays a person counter and a table with track IDs and face crops (50x50 pixels).
- **Farsi Text Support**: Renders track IDs and counter in Farsi using a provided font.
- **Image Saving**:
  - Saves an entry image for each person (bounding box crop).
  - Saves up to two random bounding box snapshots and two random skeleton snapshots per person (15% probability per frame).
- **Video Output**: Generates an output video with all visualizations (`output.mp4`).

## Prerequisites

- **Python 3.8+**
- **Dependencies** (install via `pip`):

  ```bash
  pip install ultralytics opencv-python numpy pillow arabic-reshaper python-bidi
  ```

- **YOLO Model**: Download the `yolo11x-pose.pt` model and place it in the `Data/` directory.
- **Input Video**: Place the input video (`input.mp4`) in the `Data/` directory.
- **Farsi Font**: Place a Farsi font file (`Vazirmatn-Regular.ttf`) in the `Data/` directory. You can download it from [Google Fonts](https://fonts.google.com/specimen/Vazirmatn) or another source.

## Directory Structure

```
person-tracking/
├── Data/
│   ├── yolo11x-pose.pt         # YOLOv11 pose model
│   ├── input.mp4               # Input video
│   ├── Vazirmatn-Regular.ttf   # Farsi font
├── Trail Captured/             # Output directory for images and video
│   ├── output.mp4              # Output video
│   ├── person_<ID>_entry.jpg   # Entry images
│   ├── person_<ID>_bb_random<1,2>.jpg  # Random bounding box snapshots
│   ├── person_<ID>_kp_random<1,2>.jpg  # Random skeleton snapshots
├── main.py                     # Main script
├── demo.mp4                    # Demo video of tracking
├── README.md                   # This file
```

## Usage

1. **Set up the environment**:

   - Clone the repository:

     ```bash
     git clone https://github.com/AmirSalajegheh/person_tracking.git
     cd person-tracking
     ```

   - Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```

   - Ensure the `Data/` directory contains `yolo11x-pose.pt`, `input.mp4`, and `Vazirmatn-Regular.ttf`.

2. **Run the script**:

   ```bash
   python main.py
   ```

   - The script processes `input.mp4`, displays the tracking in a window, and saves the output video and images in `Trail Captured/`.
   - Press `q` to stop the processing early.

3. **Output**:

   - **Video**: `Trail Captured/output.mp4` contains the processed video with bounding boxes, skeletons, trails, and the side panel.
   - **Images**: `Trail Captured/` contains entry images and random snapshots for each tracked person.
   - **Demo**: Watch `demo.mp4` to see the tracking in action.

## Configuration

The script includes several configurable parameters at the top of `main.py`:

- `MODEL_PATH`: Path to the YOLO model (`Data/yolo11x-pose.pt`)
- `VIDEO_PATH`: Path to the input video (`Data/input.mp4`)
- `CAPTURE_DIR`: Directory for output images and video (`Trail Captured`)
- `RANDOM_SNAPSHOT_PROB`: Probability of saving a random snapshot (0.15)
- `MAX_BB_SNAPSHOTS` / `MAX_KP_SNAPSHOTS`: Maximum number of bounding box/skeleton snapshots per person (2 each)
- `CONF_THRESHOLD`: Detection confidence threshold (0.5)
- `KEYPOINT_CONF_THRESHOLD`: Keypoint confidence threshold for skeleton lines (0.5)
- `TRAIL_MAX_POINTS`: Maximum trail points (100)
- `TRAIL_TIMEOUT`: Seconds before clearing a trail (3.0)
- `FONT_PATH`: Path to the Farsi font (`Data/Vazirmatn-Regular.ttf`)
- `SIDE_PANEL_WIDTH`: Width of the side panel (176 pixels)
- `FACE_CROP_SIZE`: Size of face crop square (50 pixels)

## Notes

- The script assumes the input video contains people (COCO class ID 0).
- The `Trail Captured/` directory is created automatically if it doesn't exist.
- Face crops are extracted using the nose keypoint and updated every frame.
- The output video has an extended width to accommodate the side panel.
- The script uses OpenCV for video processing and Pillow for Farsi text rendering.
- Ensure the Farsi font file is valid, or the script will raise an error.

## Demo

Watch the `demo.mp4` video in the repository to see the tracking system in action, including bounding boxes, skeletons, trails, and the side panel with person counter and face crops.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLOv11 model.
- [Vazirmatn Font](https://fonts.google.com/specimen/Vazirmatn) for Farsi text rendering.
