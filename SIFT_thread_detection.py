import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import threading
import queue

MIN_MATCH_COUNT = 20

# Initialize SIFT detector
def initialize_sift():
    """
    Initializes the SIFT (Scale-Invariant Feature Transform) feature detector.

    Returns:
        - detector: The SIFT feature detector initialized with custom parameters.
        
    Parameters:
        - nfeatures: The maximum number of features to retain (default: 5000).
        - contrastThreshold: The contrast threshold for feature detection (default: 0.01).
        - edgeThreshold: The edge threshold for feature detection (default: 5).
        - sigma: The sigma value for Gaussian smoothing (default: 1.2)
    """
    
    return cv2.SIFT_create(
        nfeatures=5000,
        contrastThreshold=0.01,
        edgeThreshold=5,
        sigma=1.2
    )

# Load and preprocess the reference image
def load_reference_image(image_path, detector):
    """
     Loads and processes the reference image for object detection by extracting key points and descriptors.

    Parameters:
        - image_path: Path to the reference image.
        - detector: The feature detector (e.g., SIFT) to use for keypoint detection.

    Returns:
        - trainImg: The loaded reference image in grayscale.
        - trainKP: Key points detected in the reference image.
        - trainDesc: Descriptors for the detected key points.
        - trainBorder: The border coordinates of the reference image.
    """
    trainImg = cv2.imread(image_path, 0)
    trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
    h, w = trainImg.shape
    trainBorder = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    return trainImg, trainKP, trainDesc, trainBorder

# Object detection function
def detect_object_points(query_frame, trainKP, trainDesc, trainBorder, detector, bf):
    """
    Detects the object points in the query frame by performing feature matching with the reference image.

    Parameters:
        - query_frame: The frame from the camera to detect the object in.
        - trainKP: Key points of the reference image.
        - trainDesc: Descriptors of the reference image.
        - trainBorder: The border points of the reference image.
        - detector: The feature detector (e.g., SIFT) used for detecting key points.
        - bf: The brute-force matcher for feature matching.

    Returns:
        - queryBorder: The detected object's border in the query frame, or None if no object is detected.
    """
    # Convert the frame to grayscale
    QueryImg = cv2.cvtColor(query_frame, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc = detector.detectAndCompute(QueryImg, None)

    # Perform feature matching
    matches = bf.knnMatch(queryDesc, trainDesc, k=2)

    # Apply Lowe's ratio test to filter good matches
    goodMatch = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(goodMatch) > MIN_MATCH_COUNT:
        tp, qp = [], []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32(tp), np.float32(qp)

        if tp.shape[0] >= 4 and qp.shape[0] >= 4:
            H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 5.0)
            if H is not None:
                queryBorder = cv2.perspectiveTransform(trainBorder, H)
                return np.int32(queryBorder)  # Return the points of the detected object
    return None  # Return None if no object is detected

def detection_thread_func(frame_queue, result_queue, trainKP, trainDesc, trainBorder, detector, bf, stop_event):
    """
    Runs in a separate thread to process frames, detect object points, and place results in a queue.

    Parameters:
        - frame_queue: Queue containing frames to process.
        - result_queue: Queue to place detected object points for each frame.
        - trainKP: Key points of the reference image.
        - trainDesc: Descriptors of the reference image.
        - trainBorder: Border points of the reference image.
        - detector: The feature detector (e.g., SIFT).
        - bf: The brute-force matcher for feature matching.
        - stop_event: Event used to stop the thread.

    Returns:
        - None: This function runs continuously until the stop_event is set.
    """
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=0.1)  # Add timeout to avoid blocking
            detected_points = detect_object_points(frame, trainKP, trainDesc, trainBorder, detector, bf)
            result_queue.put(detected_points)
        except queue.Empty:
            continue  # If the queue is empty, check the stop_event again
        

if __name__ == "__main__":
    detector = initialize_sift()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    trainImg, trainKP, trainDesc, trainBorder = load_reference_image("./inventions_de_humanite.jpg", detector)

    # Queues for communication between threads
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    # Event for stopping the thread
    stop_event = threading.Event()

    # Start the detection thread
    detection_thread = threading.Thread(target=detection_thread_func, args=(frame_queue, result_queue, trainKP, trainDesc, trainBorder, detector, bf, stop_event))
    detection_thread.start()
    
    trainImg1 = cv2.drawKeypoints(trainImg, trainKP, None, (255, 0, 0), 4)
    plt.imshow(trainImg1)
    plt.show()

    cam = cv2.VideoCapture(0)
    cam.open('http://10.64.47.81:8080/video')

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        # Push the frame to the queue if not full
        if not frame_queue.full():
            frame_queue.put(frame)

        # Get the result if available
        if not result_queue.empty():
            detected_points = result_queue.get()
            if detected_points is not None:
                cv2.polylines(frame, [detected_points], True, (0, 255, 0), 5)
                x, y, w, h = cv2.boundingRect(detected_points)
                centroid_x, centroid_y = int(x + w / 2), int(y + h / 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'Centroid: ({centroid_x}, {centroid_y})', (centroid_x + 10, centroid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Object detected at points: {detected_points}")
            else:
                print("No object detected.")

        cv2.imshow('result', frame)
        if cv2.waitKey(1) == ord('q'):
            stop_event.set()  # Signal the thread to stop
            break

    detection_thread.join()
    cam.release()
    cv2.destroyAllWindows()
