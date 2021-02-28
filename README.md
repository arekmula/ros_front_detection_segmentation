# ros_front_detection_segmentation
Ros node that's using [MaskRCNN](https://github.com/matterport/Mask_RCNN) and Tensorflow to detect and run segmetation to distinguish rotational fronts from transitional fronts.
The code for training the network can be found in [another repository](https://github.com/arekmula/mrcnn_instance_segmentation)

The node utilizes conda virtual environment to separate the environment variables such as Tensorflow version or
CUDA version.
## Dependencies
- ROS Noetic
- Anaconda

## Installation
- Create conda environment from environment.yml file `conda env create -f environment.yml`
- Activate environment `conda activate ros_mask_rcnn`
- Create catkin workspace with Python executable set from conda:
```
source /opt/ros/noetic/setup.bash
mkdir -p caktin_ws/src
cd catkin_ws
catkin_make -DPYTHON_EXECUTABLE=~/anaconda3/envs/ros_mask_rcnn/bin/python3.6
```
- Clone the repository
```
source devel/setup.bash
cd src
git clone https://github.com/arekmula/ros_front_detection_segmentation
cd ~/catkin_ws
catkin_make
```


## Run with

From activated conda environment run following commands (**remember to source ROS base and devel environment**):
- Setup ROS parameters:
```
rosparam set rgb_image_topic "image/topic"
rosparam set mrcnn_model_dir "path/to/model/mask_rcnn_model.h5"
rosparam set front_prediction_topic "topic/to/publish/prediction"
```

- Run with
```
rosrun front_detection_segmentation front_detection_segmentation_node.py 
```