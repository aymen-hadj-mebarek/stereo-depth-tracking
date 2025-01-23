import numpy as np
import cv2 as cv
import queue
import threading
import matplotlib.pyplot as plt

from stereo_detection_thread import detection_thread_func, initialize_sift, load_reference_image

def capture_and_calibrate(cap, rows=9, cols=7, size=20):
    """
    Captures a frame from a camera, detects a chessboard pattern, and performs camera calibration.

    Parameters:
        - cap: Video capture object.
        - rows: Number of chessboard rows (default: 9).
        - cols: Number of chessboard columns (default: 7).
        - size: Size of a chessboard square in real-world units (default: 20).

    Returns:
        - frame: Captured frame with drawn chessboard corners (if found).
        - K: Camera intrinsic matrix (or None if calibration fails).
        - distCoef: Distortion coefficients (or None if calibration fails).
        - rvecs: Rotation vectors (or None if calibration fails).
        - tvecs: Translation vectors (or None if calibration fails).
        - objectPoints: 3D chessboard points.
        - imgPoints: 2D detected chessboard corners.
        - imageSize: Image dimensions used for calibration.
    """
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objectPoints = []
    imgPoints = []

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Error: Failed to retrieve frame from camera.")

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        cornersRefined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), termCriteria)
        points = np.zeros((rows * cols, 3), np.float32)
        points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        points *= size

        objectPoints = [points]
        imgPoints = [cornersRefined]

        cv.drawChessboardCorners(frame, (cols, rows), cornersRefined, ret)

        ret, K, distCoef, rvecs, tvecs = cv.calibrateCamera(objectPoints, imgPoints, gray.shape[::-1], None, None)

        if ret:
            return frame, K, distCoef, rvecs, tvecs, objectPoints, imgPoints, gray.shape[::-1]
        else:
            return frame, None, None, None, None, objectPoints, imgPoints, gray.shape[::-1]
    else:
        return frame, None, None, None, None, objectPoints, imgPoints, gray.shape[::-1]

def capture_and_calibrate_multiple(url1, url2, rows=9, cols=6, size=20):
    """    
    Captures frames from two cameras, performs individual and stereo calibration, and calculates real-world coordinates.

    Parameters:
        - url1: Video stream URL or device index for Camera 1.
        - url2: Video stream URL or device index for Camera 2.
        - rows: Number of chessboard rows (default: 9).
        - cols: Number of chessboard columns (default: 6).
        - size: Size of a chessboard square in real-world units (default: 20).

    Returns:
        - R: Rotation matrix from stereo calibration.
        - T: Translation vector from stereo calibration.
        - E: Essential matrix from stereo calibration.
        - F: Fundamental matrix from stereo calibration.
    """
    cap1 = cv.VideoCapture(url1)
    cap2 = cv.VideoCapture(url2)

    if not cap1.isOpened() or not cap2.isOpened():
        raise RuntimeError("Error: Unable to open one or both cameras.")

    print("Starting calibration for Camera 1 and Camera 2...")

    while cap1.isOpened() and cap2.isOpened():
        # cap1.open(url1)
        # cap2.open(url2)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error: Failed to retrieve frames from cameras.")
            break

        frame1, K1, distCoef1, rvecs1, tvecs1, objectPoints1, imgPoints1, imageSize1 = capture_and_calibrate(cap1, rows, cols, size)
        frame2, K2, distCoef2, rvecs2, tvecs2, objectPoints2, imgPoints2, imageSize2 = capture_and_calibrate(cap2, rows, cols, size)

        if K1 is None or K2 is None:
            # print("Calibration failed")
            pass
        else:
            ret, K1, distCoef1, K2, distCoef2, R, T, E, F = cv.stereoCalibrate(
                objectPoints1, imgPoints1, imgPoints2, K1, distCoef1, K2, distCoef2, 
                imageSize1, criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6), flags=cv.CALIB_FIX_INTRINSIC
            )

            # print("[R]:\n", R)
            # print("[T]:\n", T)
            # print("[E]:\n", E)
            # print("[F]:\n", F)

            baseline = np.linalg.norm(T)

            if not query_frame1_queue.full():
                query_frame1_queue.put(frame1)
            if not query_frame2_queue.full():
                query_frame2_queue.put(frame2)

            if not result_queue1.empty() and not result_queue2.empty():
                detected_points1 = result_queue1.get()
                detected_points2 = result_queue2.get()

                if detected_points1 is not None and detected_points2 is not None:
                    x1, y1, w1, h1 = cv.boundingRect(detected_points1)
                    x2, y2, w2, h2 = cv.boundingRect(detected_points2)

                    centroid1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                    centroid2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))

                    disparity = abs(centroid1[0] - centroid2[0])

                    if disparity > 0:
                        f = K1[0, 0]
                        Z = (f * baseline) / disparity
                        X = (centroid1[0] - K1[0, 2]) * Z / f
                        Y = (centroid1[1] - K1[1, 2]) * Z / f

                        # Coordinates relative to the checkerboard
                        checkerboard_coords = np.dot(R, np.array([X, Y, Z])) + T.squeeze()
                        X_checker, Y_checker, Z_checker = checkerboard_coords

                        print(f"Coordinates in real world (Camera): X={X:.2f}, Y={Y:.2f}, Z={Z:.2f}")
                        print(f"Coordinates relative to checkerboard: X={X_checker:.2f}, Y={Y_checker:.2f}, Z={Z_checker:.2f}")

                    cv.polylines(frame1, [detected_points1], True, (0, 255, 0), 5)
                    cv.polylines(frame2, [detected_points2], True, (0, 255, 0), 5)
                    cv.circle(frame1, centroid1, 5, (0, 0, 255), -1)
                    cv.circle(frame2, centroid2, 5, (0, 0, 255), -1)
                    cv.putText(frame1, f'Disparity: {disparity}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                else:
                    pass
                    # print("Object not detected in one or both cameras.")

        cv.imshow('Camera 1', frame1)
        cv.imshow('Camera 2', frame2)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

    return R, T, E, F

if __name__ == "__main__":
    detector = initialize_sift()
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    trainImg, trainKP, trainDesc, trainBorder = load_reference_image("./inventions_de_humanite.jpg", detector)

    trainImg1 = cv.drawKeypoints(trainImg, trainKP, None, (255, 0, 0), 4)
    plt.imshow(trainImg1)
    plt.show()

    query_frame1_queue = queue.Queue(maxsize=1)
    query_frame2_queue = queue.Queue(maxsize=1)
    result_queue1 = queue.Queue(maxsize=1)
    result_queue2 = queue.Queue(maxsize=1)

    stop_event = threading.Event()

    detection_thread1 = threading.Thread(target=detection_thread_func, args=(query_frame1_queue, result_queue1, trainKP, trainDesc, trainBorder, detector, bf, stop_event))
    detection_thread1.start()

    detection_thread2 = threading.Thread(target=detection_thread_func, args=(query_frame2_queue, result_queue2, trainKP, trainDesc, trainBorder, detector, bf, stop_event))
    detection_thread2.start()

    url1 = 'https://192.168.15.206:8080/video'
    url2 = 'https://192.168.15.17:8080/video'

    try:
        R, T, E, F = capture_and_calibrate_multiple(url1, url2, rows=9, cols=7, size=20)
    except RuntimeError as e:
        print(str(e))
