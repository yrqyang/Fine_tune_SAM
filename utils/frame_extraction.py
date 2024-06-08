import cv2
import os

# Path to the MOV file
video_path = "../../dataset/DJI_0763.mov"
# Directory to save the frames
save_dir = "../../dataset/Galatsi"

# Create the save directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Frame counter
frame_number = 0

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Check if the frame is one of every 5 frames
            if frame_number % 5 == 0:
                # Save the frame (or you can replace this part with your processing code)
                frame_file = os.path.join(save_dir, f"frame_{int(frame_number/5)}.jpg")
                cv2.imwrite(frame_file, frame)

            # Increment frame counter
            frame_number += 1
        else:
            break

# Release the video capture object
cap.release()