# DataFlair Abandoned Object Detection - Revised
import numpy as np
import cv2
import os
from tracker import *

# Initialize Tracker
tracker = ObjectTracker()

# Location of video
file_path = 'video1.avi'

# Check if video exists
if not os.path.exists(file_path):
    print(f"Error: Video file '{file_path}' not found.")
    print("Please place the video file in the same directory as this script or update the path.")
    exit()

# Open the video
cap = cv2.VideoCapture(file_path)
if not cap.isOpened():
    print(f"Error: Could not open video file '{file_path}'.")
    exit()

# Get first frame from video to use as reference
ret, firstframe = cap.read()
if not ret:
    print("Error: Could not read the first frame from video.")
    cap.release()
    exit()

# Create a copy of the first frame if needed
firstframe_copy = firstframe.copy()

# Process first frame
firstframe_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
firstframe_blur = cv2.GaussianBlur(firstframe_gray, (3, 3), 0)

# Optional: Show the first frame
cv2.imshow("Reference Frame", firstframe)

print("Starting abandoned object detection...")
print("Press 'q' to quit")

# Reset video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video reached or error reading frame.")
        break
    
    # Process current frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (3, 3), 0)

    # Find difference between first frame and current frame
    frame_diff = cv2.absdiff(firstframe_blur, frame_blur)

    # Canny Edge Detection
    edged = cv2.Canny(frame_diff, 5, 200) 

    # Apply morphological operations to clean up edges
    kernel = np.ones((10, 10), np.uint8)
    thresh = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours of all detected objects
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    count = 0
    for c in cnts:
        contourArea = cv2.contourArea(c)
        
        # Filter contours by area to reduce noise
        if contourArea > 50 and contourArea < 10000:
            count += 1
            (x, y, w, h) = cv2.boundingRect(c)
            detections.append([x, y, w, h])

    # Update tracker with new detections
    _, abandoned_objects = tracker.update(detections)
    
    # Draw rectangle and ID over all abandoned objects
    for objects in abandoned_objects:
        _, x2, y2, w2, h2, _ = objects
        cv2.putText(frame, "Suspicious object detected", (x2, y2 - 10), 
                   cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 2)
        cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

    # Show number of abandoned objects detected
    text = f"Abandoned Objects: {len(abandoned_objects)}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Abandoned Object Detection', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(15) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Detection complete.")