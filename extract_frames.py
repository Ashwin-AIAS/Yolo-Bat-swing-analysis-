import cv2
import os
from pathlib import Path
import sys

# --- Configuration ---
# Make sure your video filename is correct!
VIDEO_PATH = Path("data/sample_videos/cricket_swing.mp4") 
OUTPUT_FOLDER = Path("data/frames_to_annotate/")
FRAMES_PER_SECOND_TO_SAVE = 5 # This is the 5 fps you wanted
# --- End Configuration ---

def main():
    # 1. Create the output folder if it doesn't exist
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    print(f"Output folder created at: {OUTPUT_FOLDER.resolve()}")

    # 2. Open the video
    if not VIDEO_PATH.exists():
        print(f"Error: Video file not found at {VIDEO_PATH}")
        print("Please make sure your video is named 'cricket_swing.mp4' inside 'data/sample_videos/'")
        sys.exit(1)
        
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"Error: Could not open video file {VIDEO_PATH}")
        sys.exit(1)

    # 3. Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        print("Warning: Could not get video FPS. Defaulting to 25.")
        fps = 25

    # Calculate which frames to save
    # (e.g., if fps=30 and we want 5, save every 30/5 = 6th frame)
    frame_interval = int(fps / FRAMES_PER_SECOND_TO_SAVE)
    if frame_interval == 0:
        frame_interval = 1 # Avoid division by zero

    print(f"Video FPS: {fps:.2f}. Saving 1 frame every {frame_interval} frames.")

    # 4. Loop through video and save frames
    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video

        # Check if this is a frame we want to save
        if frame_count % frame_interval == 0:
            # Generate a filename (e.g., frame_0001.jpg)
            filename = f"frame_{saved_frame_count:04d}.jpg"
            output_path = OUTPUT_FOLDER / filename
            
            # Save the frame
            cv2.imwrite(str(output_path), frame)
            saved_frame_count += 1

        frame_count += 1

    # 5. Clean up
    cap.release()
    print(f"Done! Saved {saved_frame_count} frames to {OUTPUT_FOLDER.resolve()}")

if __name__ == "__main__":
    main()