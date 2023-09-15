import cv2
import time
import os

# Create a VideoCapture object to access the camera (0 for default camera)
cap = cv2.VideoCapture(1)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a directory to save the images if it doesn't exist
output_directory = "D:/robot\dataset/hole/"
os.makedirs(output_directory, exist_ok=True)

# Initialize a counter to track the file names
file_counter = 1

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read a frame.")
        break

    # Get the current timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # Define the filename with a sequential number
    filename = f"{output_directory}image_{file_counter}.png"

    # Save the captured frame as a PNG image
    cv2.imwrite(filename, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(f"Image saved as {filename}")

    # Increment the file counter
    file_counter += 1

    # Wait for 1 second
    time.sleep(0.3)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
