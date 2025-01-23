import numpy as np
import cv2 as cv


def capture_and_calibrate(cap, rows=9, cols=7, size=20):
    """
    Capture frames for calibration of a single camera.

    Parameters:
        cap (cv.VideoCapture): OpenCV video capture object for the camera.
        rows (int): Number of inner corners in the checkerboard rows.
        cols (int): Number of inner corners in the checkerboard columns.
        size (float): Size of each square in real-world units (e.g., mm).

    Returns:
        frame (np.ndarray): The captured frame with chessboard corners drawn (if detected).
        K (np.ndarray): Camera matrix (or None if calibration fails).
        distCoef (np.ndarray): Distortion coefficients (or None if calibration fails).
        rvecs (list): Rotation vectors (or None if calibration fails).
        tvecs (list): Translation vectors (or None if calibration fails).
    """
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objectPoints = []  # 3D points in real-world space
    imgPoints = []  # 2D points in image plane

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Error: Failed to retrieve frame from camera.")

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        # Refine the corner locations
        cornersRefined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), termCriteria)

        # Create object points
        points = np.zeros((rows * cols, 3), np.float32)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        points *= size

        # Append the points
        objectPoints = [points]
        imgPoints = [cornersRefined]

        # Draw the detected corners
        cv.drawChessboardCorners(frame, (cols, rows), cornersRefined, ret)

        # Perform calibration immediately
        ret, K, distCoef, rvecs, tvecs = cv.calibrateCamera(objectPoints, imgPoints, gray.shape[::-1], None, None)

        if ret:
            print("Calibration successful!")
            # print("[K]:\n", K)
            # print("Distortion Coefficients:\n", distCoef)
            # print("[R]:\n", np.array(rvecs).shape)
            # print("[T]:\n", np.array(tvecs).shape)
            return frame, K, distCoef, rvecs, tvecs, objectPoints, imgPoints, gray.shape[::-1]
        else:
            print("Calibration failed.")
            return frame, None, None, None, None, objectPoints, imgPoints, gray.shape[::-1]
    else:
        # No chessboard detected
        print("Chessboard not detected.")
        return frame, None, None, None, None, objectPoints, imgPoints, gray.shape[::-1]



def capture_and_calibrate_multiple(url1, url2, rows=9, cols=6, size=20):
    """
    Calibrate two cameras simultaneously by capturing frames from both cameras.

    Parameters:
        url1 (str): URL of the first camera.
        url2 (str): URL of the second camera.
        rows (int): Number of inner corners in the checkerboard rows.
        cols (int): Number of inner corners in the checkerboard columns.
        size (float): Size of each square in real-world units (e.g., mm).

    Returns:
        (K1, distCoef1, rvecs1, tvecs1), (K2, distCoef2, rvecs2, tvecs2): Calibration results for each camera.
    """
    # Initialize video capture for both cameras
    cap1 = cv.VideoCapture(url1)
    cap2 = cv.VideoCapture(url2)

    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Error: Unable to open one or both cameras.")

    print("Starting calibration for Camera 1 and Camera 2...")

    while True:
        # Capture frames from both cameras
        cap1.open(url1)
        # cap2.open(url2)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Failed to retrieve frames from cameras.")
            break

        # Perform calibration for each camera
        frame1, K1, distCoef1, rvecs1, tvecs1, objectPoints1, imgPoints1, imageSize1 = capture_and_calibrate(cap1, rows, cols, size)
        frame2, K2, distCoef2, rvecs2, tvecs2, objectPoints2, imgPoints2, imageSize2 = capture_and_calibrate(cap2, rows, cols, size)
        
        if K1 is None or K2 is None:
            print("Calibration failed")
        else:
            ret, K1, distCoef1, K2, distCoef2, R, T, E, F = cv.stereoCalibrate(
                objectPoints1, imgPoints1, imgPoints2, K1, distCoef1, K2, distCoef2, 
                imageSize1, criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6), flags=cv.CALIB_FIX_INTRINSIC
            )
            print("[R]:\n", R)
            print("[T]:\n", T)
            print("[E]:\n", E)
            print("[F]:\n", F)

        # Resize frames to have the same display size
        window_size = (640, 480)
        frame1_resized = cv.resize(frame1, window_size)
        frame2_resized = cv.resize(frame2, window_size)
        
        # Display both frames in separate windows
        cv.imshow('Camera 1', frame1_resized)
        cv.imshow('Camera 2', frame2_resized)

        # Press 'q' to quit
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

    return R,T,E,F


# Example usage
if __name__ == "__main__":
    # Camera URLs (replace with actual URLs)
    url1 = 'https://192.168.1.71:8080/video'
    url2 = 'https://192.168.1.76:8080/video'

    try:
        R,T,E,F = capture_and_calibrate_multiple(url1, url2, rows=9, cols=7, size=20)

        print("Final Calibration Results :")
        print("[R]:\n", R)
        print("[T]:\n", T)
        print("[E]:\n", E)
        print("[F]:\n", F)
    except RuntimeError as e:
        print(str(e))
