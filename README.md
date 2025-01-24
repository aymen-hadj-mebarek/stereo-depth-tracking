# Stereo Vision Object Detection and Depth Estimation

![OpenCV](https://img.shields.io/badge/OpenCV-4.7.0%20(contrib)-green) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![License](https://img.shields.io/badge/License-MIT-orange)  
**Developed as a Final Year, Module Project for USTHB**  
**MSc in Intellignet Computer Systems**  

A real-time stereo vision system for detecting objects, calculating disparity, and estimating 3D coordinates using **SIFT feature matching**, **multi-threaded processing**, and **camera calibration**.  

---

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Academic Context](#academic-context)  
4. [File Structure](#file-structure)  
5. [Dependencies](#dependencies)  
6. [Installation](#installation)  
7. [Usage](#usage)  
8. [Calibration Methodology](#calibration-methodology)  
9. [Troubleshooting](#troubleshooting)  
10. [Contact](#contact)  

---

## Project Overview  
This project uses two cameras to:  
1. Detect objects in real-time using **SIFT (Scale-Invariant Feature Transform)**.  
2. Calculate horizontal disparity between matched objects across stereo frames.  
3. Estimate real-world 3D coordinates via **stereo camera calibration**.  
4. Perform live calibration using chessboard patterns for intrinsic/extrinsic parameters.  

**Use Cases**: Robotics, augmented reality, depth sensing, and industrial automation.  

---

## Key Features  
### 1. **Multi-Threaded Object Detection**  
- **Threading & Queues**: Parallel processing of camera frames using `threading` and `queue` modules for efficient resource utilization.  
- **SIFT Feature Matching**: Robust object detection using keypoints and descriptors with Lowe's ratio test for filtering matches.  

### 2. **Stereo Vision Pipeline**  
- **Disparity Calculation**: Horizontal pixel difference between object centroids in left/right cameras.  
- **3D Coordinate Estimation**: Converts disparity to real-world depth using camera baseline and focal length.  

### 3. **Camera Calibration**  
- **Intrinsic Calibration**: Computes camera matrix (`K`) and distortion coefficients using chessboard patterns.  
- **Stereo Calibration**: Estimates rotation (`R`), translation (`T`), essential (`E`), and fundamental (`F`) matrices for 3D reconstruction.  

### 4. **Real-Time Visualization**  
- OpenCV-based live streams with annotated object boundaries, centroids, and disparity values.  

---

## Academic Context  
This project was completed as part of the **Final Year Project** for the **Intellignet Computer Systems** program at **USTHB**.  

### Objectives  
1. Demonstrate proficiency in computer vision and real-time systems.  
2. Implement stereo vision principles for depth estimation.  
3. Address challenges in multi-threaded synchronization and camera calibration.  

## 👥 Collaborators  

| [![Aymen HM.](https://github.com/aymen-hadj-mebarek.png)](https://github.com/aymen-hadj-mebarek) | [![Nadjib L.](https://github.com/Lacomine02.png)](https://github.com/Lacomine02) | [![Yasmine A.](https://github.com/JasminCoding.png)](https://github.com/JasminCoding) | [![Ramy D.](https://github.com/Wea-boo.png)](https://github.com/Wea-boo) |  
|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|  
| **[Aymen Hadj Mebarek](https://github.com/yourusername)**                                               | **[Nadjib L.](https://github.com/collab1)**                                         | **[Yasmine A.](https://github.com/JasminCoding)**                                         | **[Ramy D.](https://github.com/Wea-boo)**                                         |  

### Advisors  
- **Mr. Abada Lyes**: Provided guidance on stereo calibration and OpenCV integration.  

---

## File Structure

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `stereo_detection_thread.py` | Main script for stereo object detection and disparity visualization. Uses multi-threading for simultaneous frame processing. |
| `stereo_depth_tracking.py`   | Extends detection to compute real-world 3D coordinates using stereo calibration results. |
| `SIFT_thread_detection.py`   | Implements SIFT-based object detection in a threaded environment. Used as a module by other scripts. |
| `live_camera_calibration.py` | Performs single-camera calibration using a live chessboard feed. |
| `stereo_calibration_live.py` | Calibrates stereo camera pairs to compute extrinsic parameters (R, T, E, F). |

---

## Dependencies
- **Python 3.7+**
- **OpenCV (contrib version)**: Includes SIFT and calibration modules.
- **NumPy**: For matrix operations and 3D coordinate calculations.
- **Matplotlib**: For visualizing reference image keypoints.

## Installation

### 1. Clone the Repository

```shell
git clone https://github.com/aymen-hadj-mebarek/stereo-depth-tracking.git
cd stereo-depth-tracking
```

### 2. Set Up a Virtual Environment (Recommended)

```shell
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```shell
pip install -r requirements.txt
```
### 4. Prepare Reference Images

- Place your reference image (e.g., `chips.jpg`) in the project root.
    
- Update the image path in `stereo_detection_thread.py` or `SIFT_thread_detection.py` if needed.
    

---

## Usage

### 1. Single-Camera Object Detection

```shell
python SIFT_thread_detection.py
```

- **Workflow**:
    
    1. Initializes SIFT detector and loads a reference image.
        
    2. Captures live frames from a camera (IP/webcam).
        
    3. Detects objects using feature matching and displays centroids.
        

### 2. Stereo Detection & Disparity Calculation

```shell
python stereo_detection_thread.py
```

- **Workflow**:
    
    1. Initializes two camera streams (update `url1` and `url2` to your IP cameras).
        
    2. Processes frames in parallel using threads.
        
    3. Computes disparity between detected objects and displays annotated streams.
        

### 3. Camera Calibration

```shell
python live_camera_calibration.py
```

- **Steps**:
    
    1. Show a chessboard (default: 9x7 inner corners) to the camera.
        
    2. The script automatically detects corners and calibrates the camera.
        

### 4. Stereo Depth Tracking

```shell
python stereo_depth_tracking.py
```

- **Workflow**:
    
    1. Calibrates both cameras using a chessboard.
        
    2. Detects objects in stereo frames and calculates 3D coordinates relative to the chessboard.
        
    3. Outputs real-world coordinates (X, Y, Z) in the console.
        

---

## Calibration Methodology

1. **Chessboard Pattern**: Uses a grid of known dimensions (e.g., 9x7 inner corners) to compute camera parameters.
    
2. **Intrinsic Parameters**:
    
    - `K` (camera matrix): Focal length and optical center.
        
    - `distCoef` (distortion coefficients): Radial and tangential distortion.
        
3. **Extrinsic Parameters**:
    
    - `R` (rotation) and `T` (translation): Spatial relationship between the two cameras.
        
4. **Stereo Calibration**: Performed using `cv.stereoCalibrate()` to align both cameras into a unified coordinate system.
    

---

## Troubleshooting

|Issue|Solution|
|---|---|
|**Camera Connection Failed**|Update IP addresses in scripts or check camera permissions.|
|**Calibration Fails**|Ensure the chessboard is fully visible and well-lit.|
|**Low Detection Accuracy**|Adjust SIFT parameters in `initialize_sift()` (e.g., `nfeatures`, `contrastThreshold`).|
|**High CPU Usage**|Reduce frame resolution or use hardware-accelerated cameras.|

---

## Future Enhancements

- **GUI Integration**: User-friendly interface for parameter tuning.
    
- **Depth Map Generation**: Generate dense disparity maps via block matching.
    
- **ROS Integration**: Publish 3D coordinates to a ROS topic for robotics applications.
    
- **Performance Optimization**: Replace SIFT with ORB for faster detection (trade-off with accuracy).

---

## Contact

For questions or feedback, contact [Your Name] at [[aymenhm34@gmail.com](mailto:aymenhm34@gmail.com)].
