# ros_front_detection_segmentation


## Dependencies
- ROS Noetic
- Anaconda
- Python 3.7
- Tensorflow 1.4.0
- Keras 2.1.2
- scikit-image 0.18.1
- scipy 1.6.1


## Installation
- Create conda environment from environment.yml file `conda env create -f environment.yml`
- Activate environment `conda activate mrcnn_ros`
- Create catkin workspace with Python executable set from conda

```
mkdir -p caktin_ws/src
cd catkin_ws
catkin_make -DPYTHON_EXECUTABLE=~/anaconda3/envs/mrcnn_ros/bin/python3.7
```
- Clone the repository
```
cd src
git clone https://github.com/arekmula/ros_front_detection_segmentation
cd ~/catkin_ws
catkin_make
```


## Run with

- Setup RGB image (640x480) topic:
```
rosparam set rgb_image_topic "image/topic"
rosparam set mrcnn_model_dir "path/to/model/mask_rcnn_model.h5"
```

- Run with
```
rosrun front_detection_segmentation front_detection_segmentation_node.py 
```