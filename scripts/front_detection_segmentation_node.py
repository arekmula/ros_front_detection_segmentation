#!usr/bin/env python3.7

# System imports
import sys
import threading

# Python packages
import cv2
import numpy as np

# ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String, Header

# ROS package imports
from front_detection_segmentation.msg import FrontPrediction

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

        # Topic name where prediction will be published
        front_prediction_topic = rospy.get_param("front_prediction_topic", default="front_prediction")
        self.prediction_pub = rospy.Publisher(front_prediction_topic, FrontPrediction, queue_size=1)

        # Topic name of 640x480 image to subscribe
        self.rgb_image_topic = rgb_image_topic

        # Should publish visualization image
        self.should_publish_visualization = rospy.get_param("visualize_front_prediction", True)
        if self.should_publish_visualization:
            self.vis_pub = rospy.Publisher("front_visualization", Image, queue_size=1)

        self.cv_bridge = CvBridge()

        # Inference config
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

                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                # Prediction on single image frame
                prediction = self.model.detect([cv_image], verbose=0)[0]

                # Publish visualization image
                if self.should_publish_visualization:
                    visualized_image = self.visualize_prediction(prediction, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
                    cv_prediction = np.zeros(shape=visualized_image.shape, dtype=np.uint8)
                    cv2.convertScaleAbs(visualized_image, cv_prediction)
                    visualization_msg = self.cv_bridge.cv2_to_imgmsg(cv_prediction, "bgr8")
                    self.vis_pub.publish(visualization_msg)

                # Build prediction message
                prediction_msg = self.build_prediction_msg(msg, prediction)
                self.prediction_pub.publish(prediction_msg)

                # Show open cv window
                # cv2.imshow("Image", cv_prediction)
                # cv2.waitKey(1)

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

    def build_prediction_msg(self, msg, prediction):
        prediction_msg = FrontPrediction()
        prediction_msg.header = msg.header

        # For each prediction ROI create prediction message
        for i, (y1, x1, y2, x2) in enumerate(prediction["rois"]):
            # Create boudning box field for current prediction ROI
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2-x1)
            prediction_msg.boxes.append(box)

            # Add class_id
            class_id = prediction['class_ids'][i]
            prediction_msg.class_ids.append(class_id)

            # Add class name
            class_name = self.class_names[class_id]
            prediction_msg.class_names.append(class_name)

            # Add prediction score
            score = prediction["scores"][i]
            prediction_msg.scores.append(score)

            # Create prediction mask for current ROI
            mask = Image()
            mask.header = msg.header
            mask.height = prediction["masks"].shape[0]
            mask.width = prediction["masks"].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            prediction_mask = (prediction["masks"][:, :, i]).astype(np.uint8)  # Convert boolean array to int array
            mask.data = (prediction_mask * 255).tobytes()
            prediction_msg.masks.append(mask)

        return prediction_msg


def main(args):
    rospy.init_node("front_detection_segmentation")
    if rospy.has_param("rgb_image_topic"):
        rgb_image_topic = rospy.get_param("rgb_image_topic")
        print(rgb_image_topic)

    detector_segmentator = DetectorSegmentator(rgb_image_topic)
    detector_segmentator.run()


if __name__ == '__main__':
    main(sys.argv)
