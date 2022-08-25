# Lawny Compute Subsystem

Lawny's computational subsystem is in charge of modelling Lawny's position and orientation in 3D space while simultaneously building a semantically informed map of the environment - using these tools the computational subsystem then executes a control loop to send position and drive commands to the embedded system.

The goal was to be able to semantically detect which portions of the stereo camera input imagery is considered mowable grass. This is stored and probablistically updated in the map so the robot could build a mowing plan for the lawn.

## Source Code

The computational subsystem is split into three folders:
* `server` - the MQTT server used to communicate with other subsystems (Mosquitto MQTT also works).
* `research` - the Jupyter notebooks for developing some of our image processing and perception pipeline.
* `vision` - the core computational application.

**For the most significant part of the code base check out the folder: [vision/vslam](vision/vslam)**

## Outcome

We were able to achieve significant progress in a relatively short period of time for our fourth year design project presentation and demonstration. We were able to implement most components of what was required to achieve the aforementioned goal. However, due to hardware and time limitations we were unable to get all components running smoothly together. As an example, we got our semantic segmentation model working and we even optimized it using various methods but it was simply to expensive to run in the main computation loop on the NVidia Jetson. Another issue we faced is that properly synchronizing and calibrating the IMX219-83 Stereo Camera was not working as well as expected so we had to deal with suboptimal results for our stereo depth estimator (distorted and/or unsychronized images from either of the two cameras leads to reduced accuracy in disparity computations).

## Improvements (To-do)

Some things to improve:
* Get all computation components running together by offloading some computation to faster hardware (maybe allow PC to run computation or AWS for production system).
* Fix drift problem and improve performance by switching VSLAM estimator to use a Kalman filter on the strongest correlated features (currently we use a heuristic gradient ascent based approach).
* Fix IMX219-83 Stereo Camera sensor camera sychronization (left and right images should be captured at the same time) to improve depth estimation accuracy.
* Possibly use more advanced state estimation methods to achieve better results for longer drives/simulations (e.g. pose graph optimization can be used to improve accuracy for when robot sees imagery it has previously seen).
