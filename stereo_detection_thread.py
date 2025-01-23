import time
import cv2
import numpy as np
import threading
import queue
from SIFT_thread_detection import initialize_sift, load_reference_image, detect_object_points, detection_thread_func
import matplotlib.pyplot as plt

# Stereo detection function using threads
def stereo_detect_threaded(query_frame1_queue, query_frame2_queue, result_queue1, result_queue2, trainKP, trainDesc, trainBorder, detector, bf, stop_event):
    """
    Perform stereo detection on two frames using threading.
    """
    thread1 = threading.Thread(
        target=detection_thread_func, 
        args=(query_frame1_queue, result_queue1, trainKP, trainDesc, trainBorder, detector, bf, stop_event)
    )
    thread2 = threading.Thread(
        target=detection_thread_func, 
        args=(query_frame2_queue, result_queue2, trainKP, trainDesc, trainBorder, detector, bf, stop_event)
    )

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()

if __name__ == "__main__":
    detector = initialize_sift()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Load reference image
    trainImg, trainKP, trainDesc, trainBorder = load_reference_image("chips.jpg", detector)

    # Create queues for frames and results
    query_frame1_queue = queue.Queue(maxsize=1)
    query_frame2_queue = queue.Queue(maxsize=1)
    result_queue1 = queue.Queue(maxsize=1)
    result_queue2 = queue.Queue(maxsize=1)

    # Event for stopping threads
    stop_event = threading.Event()

 # Start the detection thread
    detection_thread1 = threading.Thread(target=detection_thread_func, args=(query_frame1_queue, result_queue1, trainKP, trainDesc, trainBorder, detector, bf, stop_event))
    detection_thread1.start()
    
    detection_thread2 = threading.Thread(target=detection_thread_func, args=(query_frame2_queue, result_queue2, trainKP, trainDesc, trainBorder, detector, bf, stop_event))
    detection_thread2.start()
    
    trainImg1 = cv2.drawKeypoints(trainImg, trainKP, None, (255, 0, 0), 4)
    plt.imshow(trainImg1)
    plt.show()
    
    url1 = 'https://192.168.1.71:8080/video'
    url2 = 'https://192.168.1.76:8080/video'

    cam1 = cv2.VideoCapture(url1)
    cam2 = cv2.VideoCapture(url2)
    

    while True:
        cam1.open(url1)
        cam2.open(url2)

        ret1, frame1 = cam1.read()
        ret2, frame2 = cam2.read()
        
        # cv2.flip(frame1, 1, frame1)
        # cv2.flip(frame2, 1, frame2)

        if not ret1 or not ret2:
            print("Failed to grab frames from one or both cameras.")
            break

        # Add frames to queues
        if not query_frame1_queue.full():
            query_frame1_queue.put(frame1)
        if not query_frame2_queue.full():
            query_frame2_queue.put(frame2)

        # Get results from queues
        if not result_queue1.empty() and not result_queue2.empty():
            detected_points1 = result_queue1.get()
            detected_points2 = result_queue2.get()

            if detected_points1 is not None and detected_points2 is not None:
                # Calculate disparity and centroids
                x1, y1, w1, h1 = cv2.boundingRect(detected_points1)
                x2, y2, w2, h2 = cv2.boundingRect(detected_points2)

                centroid1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                centroid2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                disparity = abs(centroid1[0] - centroid2[0])

                # Draw detected objects and centroids
                cv2.polylines(frame1, [detected_points1], True, (0, 255, 0), 5)
                cv2.polylines(frame2, [detected_points2], True, (0, 255, 0), 5)
                cv2.circle(frame1, centroid1, 5, (0, 0, 255), -1)
                cv2.circle(frame2, centroid2, 5, (0, 0, 255), -1)
                cv2.putText(frame1, f'Disparity: {disparity}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                print(f"Object position in Camera 1: ({centroid1[0]}, {centroid1[1]})")
                print(f"Object position in Camera 2: ({centroid2[0]}, {centroid2[1]})")
                print(f"Disparity: {disparity}")
            else:
                print("Object not detected in one or both cameras.")

        # Resize frames to have the same display size
        window_size = (640, 480)
        frame1_resized = cv2.resize(frame1, window_size)
        frame2_resized = cv2.resize(frame2, window_size)

        # Display the results
        cv2.imshow('Camera 1', frame1_resized)
        cv2.imshow('Camera 2', frame2_resized)

        if cv2.waitKey(10) == ord('q'):
            stop_event.set()  # Signal threads to stop
            break

    cam1.release()
    cam2.release()
    cv2.destroyWindow('Camera 1')
    cv2.destroyWindow('Camera 2')
    cv2.destroyAllWindows()
