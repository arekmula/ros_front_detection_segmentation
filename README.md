# ros_front_detection_segmentation


## Dependencies
- ROS Noetic

## Installation
```
mkdir -p caktin_ws/src
cd catkin_ws
catkin_make
cd src
git clone https://github.com/arekmula/ros_front_detection_segmentation
cd ~/catkin_ws
catkin_make
```

## Run with

- Setup RGB image (640x480) topic:
```
rosparam set rgb_image_topic "image/topic"
```

- Run with
```
rosrun front_detection_segmentation front_detection_segmentation.py ```