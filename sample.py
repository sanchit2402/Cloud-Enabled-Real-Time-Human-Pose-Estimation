import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks

# Specify video source (0 for webcam or file path for video)
video_source = 0;  # Replace with 0 for webcam

# Check if the input is a webcam or a video file
if isinstance(video_source, str) and os.path.isfile(video_source):
    cap = cv2.VideoCapture(video_source)  # For video file
else:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For webcam (Windows)
    # For Linux/Mac, use: cap = cv2.VideoCapture(0)

# Check if the video capture opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Initialize MediaPipe Pose
with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False, 
                  min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video capture ended or cannot read frame.")
            break

        # Flip frame horizontally for a mirror-like effect (optional)
        frame = cv2.flip(frame, 1) if video_source == 0 else frame  # Flip only for webcam

        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe Pose
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Landmarks
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Connections
            )

        # Resize frame to fit the screen (optional)
        display_frame = cv2.resize(frame, (960, 540))  # Adjust width and height as needed

        # Display the processed frame
        cv2.imshow('Pose Estimation', display_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
