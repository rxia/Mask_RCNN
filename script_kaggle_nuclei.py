from config import Config
import utils
import model as modellib
import visualize
from model import log
import cv2

import os
import csv
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
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 foreground

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50


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
        mask = (mask>0.5).astype('uint8')
        # mask = (np.sum(mask, axis=2) > 0.5).astype('uint8')
        if len(mask.shape)==2:
            mask = mask[:,:,None]
        # class_ids = np.array([class_ids[0]]).astype('int32')
        return mask, class_ids


n_img_train = 500
n_img_val = 100

keys_data = list(data_train.keys())
keys_random = np.random.choice(keys_data,n_img_train+n_img_val,replace=False)
keys_train = keys_random[range(n_img_train)]
keys_val = keys_random[range(n_img_train,n_img_train+n_img_val,1)]

dataset_train_all = NucleiDataset()
dataset_train_all.add_data_dict(data_train)
dataset_train_all.prepare()

dataset_train = NucleiDataset()
dataset_train.add_data_dict({key:data_train[key] for key in keys_train})
dataset_train.prepare()

dataset_val = NucleiDataset()
dataset_val.add_data_dict({key:data_train[key] for key in keys_val})
dataset_val.prepare()

dataset_test = NucleiDataset()
dataset_test.add_data_dict(data_test)
dataset_test.prepare()


# fig, ax = plt.subplots(1,2)
# i = 9
# plt.subplot(ax[0])
# plt.imshow(dataset_train.load_image(i))
# plt.subplot(ax[1])
# plt.imshow(dataset_train.load_mask(i)[0][:,:,0])


model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)
init_with = "last"  # imagenet, coco, or last
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


# ## Training


model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=10,
            layers='heads')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=30,
            layers='4+')

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=30,
            layers="all")


# ## Validate

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


##  Test on a random image
image_id = np.random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())
##


## Evaluation

# Compute VOC-Style mAP @ IoU=0.5
image_ids = np.random.choice(dataset_val.image_ids, 100)
APs = []
P_Kaggles = []
iou_thresholds = np.arange(0.5,0.95,0.05)
for i in iou_thresholds:
    APs_i = []
    P_Kaggle_i = []
    print('threshold: {}'.format(i))
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config,
                                                                                  image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precision_kaggle, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'],
                                                             iou_threshold = i)
        APs_i.append(AP)
        P_Kaggle_i.append(precision_kaggle)
    APs.append(APs_i)
    P_Kaggles.append(P_Kaggle_i)

print("mAP: ", np.mean(np.array(APs),axis=1))
print("Precision Kaggle: ", np.mean(np.array(P_Kaggles),axis=1))


##
def rle_encoding(mask):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    non_zeros = np.where(mask.T.flatten() > 0)[0] + 1    # .T sets Fortran order down-then-right
    run_lengths = []
    for i in range(len(non_zeros)):
        if i==0 or non_zeros[i]-non_zeros[i-1] > 1:
            start = non_zeros[i]
            count = 1
            run_lengths.append(start)
            run_lengths.append(count)
        elif non_zeros[i]-non_zeros[i-1] == 1:
            run_lengths[-1] += 1
    return run_lengths


test_img_keys = list(data_test.keys())
images = [data_test[key]['image'] for key in test_img_keys]
results = []
for image in images:
    results.append(model.detect([image], verbose=1))

mask_ImageId = []
mask_EncodedPixels = []
for i in range(len(results)):
    masks_i = results[i][0]['masks']
    for j in range(results[i][0]['masks'].shape[2]):
        sum_masks_i = np.sum(masks_i, axis=2)
        mask = masks_i[:,:,j]
        mask[sum_masks_i>1] = 0
        masks_i[:,:,j] = mask
        if np.sum(mask)>1:
            mask_ImageId.append(test_img_keys[i])
            recoded_mask = rle_encoding(mask)
            mask_EncodedPixels.append(recoded_mask)
masks = {'ImageId':mask_ImageId,'EncodedPixels':mask_EncodedPixels}

with open("test.csv","w") as f:
    f.write(",".join(masks.keys()) + "\n")
    for i in range(len(masks['ImageId'])):
        f.write(masks['ImageId'][i] + ',')
        f.write(" ".join([str(n) for n in masks['EncodedPixels'][i]]) + "\n")
    f.close()

with open("test.csv") as f:
    test = f.read()
    for row in test:
        print(row)
    f.close()

##
for i in range(len(results)):
    h_fig, h_axes = plt.subplots(1, 2, sharex='all', sharey='all', figsize=[12,8])
    plt.axes(h_axes[0])
    h_axes[0].axis('off')
    plt.imshow(images[i])
    visualize.display_instances(images[i], results[i][0]['rois'], results[i][0]['masks'], results[i][0]['class_ids'],
                            dataset_test.class_names, results[i][0]['scores'], ax=h_axes[1])
    plt.savefig('./results/test1_0314model/{}.png'.format(i))
    plt.close()


##

results_train = []
for i in range(len(keys_data)):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_train_all, inference_config,
                                                                                       i, use_mini_mask=False)
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    h_fig, h_axes = plt.subplots(1, 3, sharex='all', sharey='all', figsize=[18,8])
    plt.axes(h_axes[0])
    h_axes[0].axis('off')
    plt.imshow(original_image)

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset_train_all.class_names, ax=h_axes[1])

    results_train.append(model.detect([original_image], verbose=1))
    visualize.display_instances(original_image, results_train[i][0]['rois'], results_train[i][0]['masks'], results_train[i][0]['class_ids'],
                                dataset_train_all.class_names, results_train[i][0]['scores'], ax=h_axes[2])
    plt.savefig('./results/train1_0314model/{}.png'.format(i))
    plt.close()
##