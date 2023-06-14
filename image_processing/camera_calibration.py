import numpy as np
import cv2

# Prepare object points (assuming a chessboard pattern with 9x6 corners)
object_points = np.zeros((6 * 9, 3), np.float32)
object_points[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points
object_points_list = []  # 3D points in real world space
image_points_list = []  # 2D points in image plane

# Create video capture object
cap = cv2.VideoCapture(0)

# Define termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Counter for image filenames
counter = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If corners are found, refine the corners and draw them on the frame
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, (9, 6), corners, ret)
        image_points_list.append(corners)
        object_points_list.append(object_points)

    # Save the frame before calibration
    cv2.imwrite(f'before_calibration_{counter}.png', frame)

    # Display the frame
    cv2.imshow('Camera Calibration', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment the counter
    counter += 1

# Perform camera calibration
ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv2.calibrateCamera(
    object_points_list, image_points_list, gray.shape[::-1], None, None)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Undistort the frame
    undistorted_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients)

    # Save the frame after calibration
    cv2.imwrite(f'after_calibration_{counter}.png', undistorted_frame)

    # Display the undistorted frame
    cv2.imshow('Camera Calibration (Undistorted)', undistorted_frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Increment the counter
    counter += 1

# Release the capture object and destroy windows
cap.release()
cv2.destroyAllWindows()

# Print the calibration results
print("Camera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(distortion_coefficients)
