# Gaze-Tracker
A Deep learning based Gaze Tracker using a Hourglass Convolutional Neural Network. 

#### Dependencies
- Pytorch 0.4
- dlib

#### Pipeline
- Face detection and alignment using Dlib
- Crop eyes and feed to CNN as input
- CNN output heatmaps corresponding to salient keypoints
- Convert highest intensity on heatmap to keypoints using OpenCV

#### Real Time Demo
- python live.py

#### Example
-  https://www.youtube.com/watch?v=TwREBRt_CfA
