import cv2
import numpy as np

# Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Flag to indicate if a face has been detected
face_detected = False

while True:
    # Read the current frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Set the flag to indicate a face has been detected
        face_detected = True

    # Display the frame with face detections
    cv2.imshow('Face Recognition', frame)

    # Save the frame as an image if a face has been detected
    if face_detected:
        cv2.imwrite('tracked_face.jpg', frame)
        face_detected = False  # Reset the flag

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
