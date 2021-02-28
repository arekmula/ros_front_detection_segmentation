#!usr/bin/env python3.7

# System imports
import sys
import threading

# Python packages
import cv2
import numpy as np

# ROS imports
import rospy
from std_msgs.msg import String, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Mask RCNN
from mask_rcnn import dataset
from mask_rcnn import model as modellib
from mask_rcnn import utils
from mask_rcnn import visualize

CLASS_NAMES = ["BG", "rot_front", "trans_front"]

# TODO: Try to run with CUDA -> point to cuda tool kit and cudnn from conda environment


class InferenceConfig(dataset.FrontDetectionConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    DETECTION_MIN_CONFIDENCE = 0.95


class DetectorSegmentator:

    def __init__(self, rgb_image_topic):

        self.rgb_image_topic = rgb_image_topic

        self.cv_bridge = CvBridge()
        config = InferenceConfig()
        config.display()

        # Create model object in inference mode
        self.model = modellib.MaskRCNN(mode="inference", model_dir="", config=config)

        # Load trained weights
        model_path = rospy.get_param("mrcnn_model_dir", None)
        self.model.load_weights(model_path, by_name=True)

        # Load class names and colors
        self.class_names = CLASS_NAMES
        self.class_colors = visualize.random_colors(len(self.class_names))

        # Last input message and message lock
        self.last_msg = None
        self.msg_lock = threading.Lock()

    def run(self):
        image_sub = rospy.Subscriber(self.rgb_image_topic, data_class=Image, callback=self.image_callback,
                                     queue_size=1, buff_size=2 ** 24)

        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            # Wait for new image to be acquired
            if self.msg_lock.acquire(False):
                msg = self.last_msg
                self.last_msg = None
                self.msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                try:
                    cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
                except CvBridgeError as e:
                    print(e)

                # Prediction on single image frame
                prediction = self.model.detect([cv_image], verbose=0)[0]

                visualized_image = self.visualize_prediction(prediction, cv_image)
                cv_prediction = np.zeros(shape=visualized_image.shape, dtype=np.uint8)
                cv2.convertScaleAbs(visualized_image, cv_prediction)

                cv2.imshow("Image", cv_prediction)
                cv2.waitKey(1)

            rate.sleep()

    def image_callback(self, data):
        rospy.logdebug("Get input image")

        if self.msg_lock.acquire(False):
            self.last_msg = data
            self.msg_lock.release()

    def visualize_prediction(self, prediction, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, prediction['rois'], prediction['masks'],
                                    prediction['class_ids'], self.class_names,
                                    prediction['scores'], ax=axes,
                                    )
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))
        return result


def main(args):
    rospy.init_node("front_detection_segmentation")
    if rospy.has_param("rgb_image_topic"):
        rgb_image_topic = rospy.get_param("rgb_image_topic")
        print(rgb_image_topic)

    detector_segmentator = DetectorSegmentator(rgb_image_topic)
    detector_segmentator.run()


if __name__ == '__main__':
    main(sys.argv)
