import json
import os
import numpy as np
from PIL import Image, ImageDraw

from mask_rcnn import utils
from mask_rcnn.config import Config


class FrontHandlerConfig(Config):
    """
    Configuration for training on the fronts and handlers dataset.
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


class CocoLikeDataset(utils.Dataset):
    """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
        See http://cocodataset.org/#home for more information.
    """

    def load_data(self, annotation_json, images_dir):
        """
        Load the coco-like dataset from json
        :param annotation_json: The path to the coco annotations json file
        :param images_dir: The directory holding the images referred to by the json file
        :return:
        """
        # Load json from file
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()

        # Add the class names using the base method from utils.Dataset
        source_name = "handlers_front"  # Name of your dataset
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            # if class_id < 1:
            #     print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
            #         class_name))
            #     return

            self.add_class(source_name, class_id, class_name)

        # Get all annotations
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations.keys():
                annotations[image_id] = []
            annotations[image_id].append(annotation)

        # Get all images and add them to the dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                image_annotations = annotations[image_id]

                # Add the image using the base method from utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )

    def load_mask(self, image_id):
        """
        Load instance masks for the given image.
        MaskRCNN expects masks in the form of a bitmap [height, width, instances]
        :param image_id: The id of the image to load masks for
        :return masks: A bool array of shape [height, width, instance count] with
                one mask per instance.
        :return class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)

        return mask, class_ids