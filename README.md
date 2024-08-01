

# Vehicle and Bicycle Counting System

# Overview

This project is a real-time system for counting vehicles and bicycles using computer vision and object detection. It leverages a pre-trained YOLOv5 model to detect and track cars and bicycles in video feeds. The system counts how many cars and bicycles pass through predefined areas on the left and right sides of the frame.

# Features
Real-time Detection: Detects cars and bicycles in live video streams.
Tracking: Tracks detected objects across frames to ensure accurate counting.
Area-Based Counting: Counts vehicles and bicycles based on their entry into specific regions of the frame.
Visual Feedback: Provides real-time visual feedback with bounding boxes and counts displayed on the video feed.

# Requirements
Python 3.x

PyTorch

OpenCV

NumPy

YOLOv5 pre-trained model

# Installation

Clone the Repository:
Clone this repository to your local machine.

Install Dependencies:
Install the necessary Python packages.

Download YOLOv5 Model:
The YOLOv5 model will be automatically downloaded when running the script.

# Usage
Place your video file in the project directory.
Run the script to start the vehicle and bicycle counting process.
The system will display the video feed with real-time counts and object tracking.

# Configuration
Defining Areas: Modify the predefined areas (left and right) in the code to match your specific requirements for counting regions.

# Example
The output of the system includes real-time bounding boxes around detected vehicles and bicycles, along with counts for each type of object moving through the designated areas.
