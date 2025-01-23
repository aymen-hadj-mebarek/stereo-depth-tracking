import numpy as np
import cv2 as cv

def capture_and_calibrate(url, rows=9, cols=7, size=20):
    """
    Capture frames one by one, calibrate the camera immediately upon detecting a chessboard.

    Parameters:
        url (str): Camera stream URL.
        rows (int): Number of inner corners in the checkerboard rows.
        cols (int): Number of inner corners in the checkerboard columns.
        size (float): Size of each square in real-world units (e.g., mm).

    Returns:
        K (np.ndarray): Camera matrix.
        distCoef (np.ndarray): Distortion coefficients.
        rvecs (list): Rotation vectors.
        tvecs (list): Translation vectors.
    """
    cap = cv.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("Error: Unable to open the camera.")

    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objectPoints = []  # 3D points in real-world space
    imgPoints = []  # 2D points in image plane

    while True:
        cap.open(url)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Error: Failed to retrieve frame from the camera.")

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
            if len(imgPoints) >= 1:
                ret, K, distCoef, rvecs, tvecs = cv.calibrateCamera(objectPoints, imgPoints, gray.shape[::-1], None, None)

                if ret:
                    print("Calibration successful for current frame!")
                    print("[K]:\n", K)
                    print("Distortion Coefficients:\n", distCoef)
                    print("[R]:\n", np.array(rvecs).shape)
                    print("[T]:\n", np.array(tvecs).shape)

                else:
                    print("Calibration failed for current frame.")

        # Display the captured frame
        cv.putText(frame, "Press 'Q' to quit", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv.imshow('Camera Calibration', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

    if len(imgPoints) > 0:
        return K, distCoef, rvecs, tvecs
    else:
        raise RuntimeError("Error: No valid frames captured for calibration.")

# Example usage
if __name__ == "__main__":
    # Camera URL (replace with actual URL)
    url = 'https://10.20.3.147:8080/video'

    # Perform live calibration
    try:
        K, distCoef, rvecs, tvecs = capture_and_calibrate(url, rows=9, cols=7, size=20)
        print("Final Calibration Results:")
        print("[K]:\n", K)
        print("Distortion Coefficients:\n", distCoef)
        print("[R]:\n", np.array(rvecs).shape)
        print("[T]:\n", np.array(tvecs).shape)
    except RuntimeError as e:
        print(str(e))