# Image-based Human Body 2D Pose Estimation

Multi-person human pose estimation is defined as the task of detecting the poses (or keypoints) of all people from an input image.

Existing approaches can be categorized into top-down and bottom-up approaches.

Top-down methods (e.g. deeppose) divide the task into two stages: human detection and pose estimation. They perform human detection first, followed by single-person pose estimation given human bounding boxes.

Bottom-up approaches (e.g. AE) first detect all the keypoints and then group/associate them into person instances.

## Top-down heatmap-based pose estimation

Top-down methods divide the task into two stages: human detection and pose estimation.

They perform human detection first, followed by single-person pose estimation given human bounding boxes.
Instead of estimating keypoint coordinates directly, the pose estimator will produce heatmaps which represent the
likelihood of being a keypoint.

Various neural network models have been proposed for better performance.
The popular ones include stacked hourglass networks, and HRNet.
