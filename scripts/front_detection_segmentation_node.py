#!usr/bin/env python3.7

import sys

import cv2

# ROS imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Mask RCNN
from mask_rcnn import coco
from mask_rcnn import model as modellib
from mask_rcnn import utils

print("Python version")
print (sys.version)

class InferenceConfig(coco.FrontHandlerConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    DETECTION_MIN_CONFIDENCE = 0.5


class DetectorSegmentator:

    def __init__(self, rgb_image_topic):
        self.image_sub = rospy.Subscriber(rgb_image_topic, data_class=Image, callback=self.image_callback, queue_size=1,
                                          buff_size=2**24)
        self.cv_bridge = CvBridge()
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode
        self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)

        # Load trained weights
        model_path = rospy.get_param("mrcnn_model_path", None)

    def image_callback(self, data):
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Image", cv_image)
        cv2.waitKey(1)


def main(args):
    rospy.init_node("front_detection_segmentation")
    if rospy.has_param("rgb_image_topic"):
        rgb_image_topic = rospy.get_param("rgb_image_topic")
        print(rgb_image_topic)

    detector_segmentator = DetectorSegmentator(rgb_image_topic)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutdown")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv) 
