import json
import os
import numpy as np
from PIL import Image, ImageDraw

from mask_rcnn import utils
from mask_rcnn.config import Config


class FrontDetectionConfig(Config):
    """
    Configuration for training on the dataset used to detect fronts and run segmentation on them to distinguish
    rotational fronts from transitional fronts.
    Derives from the base Config class and overrides values specific
    to the fronts and handlers dataset
    """
    # Give the configuration a recognizable name
    NAME = "front_mask_classify"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 (__ignore__ + trans_front + rot_front)

    # All of our training images are 640x480
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 768

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 181

    DETECTION_MIN_CONFIDENCE = 0.6
