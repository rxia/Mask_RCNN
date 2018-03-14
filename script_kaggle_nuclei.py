from config import Config
import utils
import model as modellib
import visualize
from model import log
import cv2

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


with open('./data/data_train.pickle', 'rb') as f:
    data_train = pickle.load(f)
with open('./data/data_test.pickle', 'rb') as f:
    data_test = pickle.load(f)


class NucleiConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 foreground

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 64

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = NucleiConfig()
config.display()


class NucleiDataset(utils.Dataset):

    def add_data_dict(self, data_dict):
        self.data_dict = data_dict
        self.add_class("nuclei", 1, "regular")
        for i, key in enumerate(data_dict):
            self.add_image('nuclei', i, key)

    def load_image(self, image_id):
        img = self.data_dict[self.image_info[image_id]['path']]['image']
        old_size = img.shape[:2]  # old_size is in (height, width) format
        ratio = float(config.IMAGE_MAX_DIM) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = config.IMAGE_MAX_DIM - new_size[1]
        delta_h = config.IMAGE_MAX_DIM - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_img = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        return new_img

    def load_mask(self, image_id):
        mask_2D = self.data_dict[self.image_info[image_id]['path']]['mask']
        label_unique = np.unique(mask_2D)
        label_unique = label_unique[label_unique > 0]
        mask_3D = np.stack([mask_2D==label for label in label_unique], axis=2).astype('uint8')
        class_ids = np.ones(len(label_unique), dtype=np.int32)

        old_size = mask_3D.shape[:2]  # old_size is in (height, width) format
        ratio = float(config.IMAGE_MAX_DIM) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(mask_3D, (new_size[1], new_size[0]))
        delta_w = config.IMAGE_MAX_DIM - new_size[1]
        delta_h = config.IMAGE_MAX_DIM - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0]
        mask = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        mask = (np.sum(mask,axis=2)>0.5).astype('uint8')
        if len(mask.shape)==2:
            mask = mask[:,:,None]
        class_ids = np.array([class_ids[0]]).astype('int32')
        return mask, class_ids


n_img_train = 100
n_img_val = 10

keys_data = list(data_train.keys())
keys_train = np.random.choice(keys_data,n_img_train,replace=False)
keys_val = np.random.choice(keys_data,n_img_val,replace=False)

dataset_train = NucleiDataset()
dataset_train.add_data_dict({key:data_train[key] for key in keys_train})
dataset_train.prepare()

dataset_val = NucleiDataset()
dataset_val.add_data_dict({key:data_train[key] for key in keys_val})
dataset_val.prepare()


fig, ax = plt.subplots(1,2)
i = 9
plt.subplot(ax[0])
plt.imshow(dataset_train.load_image(i))
plt.subplot(ax[1])
plt.imshow(dataset_train.load_mask(i)[0][:,:,0])


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)





model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all")