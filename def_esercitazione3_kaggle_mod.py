# -*- coding: utf-8 -*-

# ATTENZIONE: model.train(dataset_train, dataset_train,
# model_path = os.path.join(MODEL_DIR, 'mask_rcnn_pneumonia_0002.weights.h5')

"""
RSNA Pneumonia Detection - Mask R-CNN
Using mrcnn package + TensorFlow 2.19

pip install matplotlib
pip install opencv-python==4.10.0.82
pip install pydicom
 pip install scikit-image
apt-get update && apt-get install -y libgl1 libglib2.0-0
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_GPU_GARBAGE_COLLECTION'] = '0'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'  # optional
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#
# os.environ['CUDA_HOME'] = '/home/sandrina/local/cuda/cuda-12.2'
# os.environ['LD_LIBRARY_PATH'] = '/home/sandrina/local/cuda/cuda-12.2/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
# os.environ['PATH'] = '/home/sandrina/local/cuda/cuda-12.2/bin:' + os.environ.get('PATH', '')

import ctypes.util

print(ctypes.util.find_library('cudart'))

# import sys
# stderr_fileno = sys.stderr
# sys.stderr = open(os.devnull, 'w')


# TensorFlow 2.19 GPU setup
import tensorflow as tf
print(tf.__version__)
#tf.config.run_functions_eagerly(True)  # forces TF to run functions eagerly
#sys.stderr = stderr_fileno

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(tf.config.list_physical_devices('GPU'))
        tf.config.experimental.set_memory_growth(gpu, True)

else:
    print("No GPU device found")

print(tf.executing_eagerly())

#input()

import random
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pydicom
from tqdm import tqdm

# Mask R-CNN
from mrcnn.config import (Config)
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize
from mrcnn.visualize import display_instances

HOST_UID = 1000
HOST_GID = 1000

# -----------------------------
# DATA and MODEL dirs
# -----------------------------
#DATA_DIR = '/home/sandrina/DL-AI_STUFF/Insegnamenti_LMDATA_PhD/PYCHARM_NOTEBOOKS/data/rsna-pneumonia-detection-challenge'
DATA_DIR = '/workspace/data/rsna-pneumonia-detection-challenge'
MODEL_DIR = os.path.join(DATA_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)

train_dicom_dir = os.path.join(DATA_DIR, 'stage_2_train_images')
#train_dicom_dir = os.path.join(DATA_DIR, 'train_small')
test_dicom_dir = os.path.join(DATA_DIR, 'stage_2_test_images')
annotations_fp = os.path.join(DATA_DIR, 'stage_2_train_labels.csv')
#annotations_fp = os.path.join(DATA_DIR, 'train_labels_small.csv')

# -----------------------------
# Helper functions
# -----------------------------

def get_colors_for_class_ids(class_ids):
    """Generate a consistent color for each class ID."""
    colors = []
    for cid in class_ids:
        random.seed(int(cid))   # same class → same color
        colors.append(np.random.rand(3,))
    return colors

def get_dicom_fps(dicom_dir):
    return list(set(glob.glob(os.path.join(dicom_dir, '*.dcm'))))


def parse_dataset(dicom_dir, anns):
    image_fps = get_dicom_fps(dicom_dir)
    image_annotations = {fp: [] for fp in image_fps}
    for index, row in anns.iterrows():
        fp = os.path.join(dicom_dir, row['patientId'] + '.dcm')
        image_annotations[fp].append(row)
    return image_fps, image_annotations


# -----------------------------
# Config class
# -----------------------------
class DetectorConfig(Config):
    NAME = 'pneumonia'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    NUM_CLASSES = 1 + 1
    BACKBONE = 'resnet50'
    #BACKBONE_STRIDES = [4, 8, 16, 32, 64]  # typical ResNet50 pyramid strides
    RPN_ANCHOR_SCALES = (32, 64)
    IMAGE_MIN_DIM = 128#64
    IMAGE_MAX_DIM = 128#64
    TRAIN_ROIS_PER_IMAGE = 16
    MAX_GT_INSTANCES = 3
    DETECTION_MAX_INSTANCES = 3
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.1
    RPN_TRAIN_ANCHORS_PER_IMAGE = 16
    STEPS_PER_EPOCH = 100
    POST_NMS_ROIS_TRAINING = 200
    TOP_DOWN_PYRAMID_SIZE = 32
    LEARNING_RATE = 0.001
    #USE_RPN_ROIS = True

    def BACKBONE_SHAPES(self):
        """Compute width and height of each stage of the backbone pyramid."""
        return np.array(
            [[int(np.ceil(self.IMAGE_MAX_DIM / stride)),
              int(np.ceil(self.IMAGE_MAX_DIM / stride))]
             for stride in self.BACKBONE_STRIDES]
        )

config = DetectorConfig()
config.display()


# -----------------------------
# Dataset class
# -----------------------------model = MaskRCNNModel(keras_model, config)

class DetectorDataset(utils.Dataset):
    def __init__(self, image_fps, image_annotations, orig_height, orig_width):
        super().__init__()
        self.add_class('pneumonia', 1, 'Lung Opacity')
        for i, fp in enumerate(image_fps):
            annotations = image_annotations[fp]
            self.add_image('pneumonia', image_id=i, path=fp,
                           annotations=annotations, orig_height=orig_height, orig_width=orig_width)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

    def load_image(self, image_id):
        fp = self.image_info[image_id]['path']
        ds = pydicom.dcmread(fp)
        image = ds.pixel_array
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        count = len(annotations)
        if count == 0:
            mask = np.zeros((info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros((info['orig_height'], info['orig_width'], count), dtype=np.uint8)
            class_ids = np.zeros((count,), dtype=np.int32)
            for i, a in enumerate(annotations):
                if a['Target'] == 1:
                    x, y, w, h = int(a['x']), int(a['y']), int(a['width']), int(a['height'])
                    mask_instance = mask[:, :, i].copy()
                    cv2.rectangle(mask_instance, (x, y), (x + w, y + h), 255, -1)
                    mask[:, :, i] = mask_instance
                    class_ids[i] = 1
        return mask.astype(bool), class_ids.astype(np.int32)


# -----------------------------
# Load annotations and parse dataset
# -----------------------------
anns = pd.read_csv(annotations_fp)
print(anns.head(6))# training dataset
image_fps, image_annotations = parse_dataset(train_dicom_dir, anns)

# Original DICOM image size
ORIG_SIZE = 1024

# -----------------------------
# Split train/val
# -----------------------------
image_fps_list = list(image_fps[:1000])  # for demo purposes
random.seed(42)
random.shuffle(image_fps_list)
split_index = int(0.9 * len(image_fps_list))
image_fps_train = image_fps_list[:split_index]
image_fps_val = image_fps_list[split_index:]

dataset_train = DetectorDataset(image_fps_train, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_train.prepare()
dataset_val = DetectorDataset(image_fps_val, image_annotations, ORIG_SIZE, ORIG_SIZE)
dataset_val.prepare()

# -----------------------------
# Visualize images
# -----------------------------

# Choose dataset
dataset = dataset_train  # train set

# Display 4 random images from train set
for i in range(10):
    image_id = random.choice(dataset.image_ids)
    # Load image and ground truth data
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(
        dataset, config, image_id
    )

    # Display original image with GT boxes and masks
    fig1=plt.figure(figsize=(18,9))
    visualize.display_instances(
        original_image, gt_bbox, gt_mask, gt_class_id,
        dataset.class_names,
        colors=get_colors_for_class_ids(gt_class_id))

    # ---- Save & close ----
    plt.tight_layout()
    fig_path=DATA_DIR+'/'+str(image_id)+'.png'
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    # Change ownership to host user
    os.chown(fig_path, HOST_UID, HOST_GID)
    # Optional: make sure read/write permissions are set
    os.chmod(fig_path, 0o666)  # rw-rw-rw-

    plt.show()
    plt.close(fig1)

# -----------------------------
# Check dataset train
# -----------------------------
# for image_id in dataset_train.image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#
#     print(f"Image ID {image_id}: shape={image.shape}, dtype={image.dtype}")
#     print(f"  Masks: shape={mask.shape}, dtype={mask.dtype}, unique class IDs: {np.unique(class_ids)}")
#
#     # Optional: skip images without any annotations
#     if np.all(class_ids == 0):
#         print("  --> This image has no positive instances")
#
#     input()

# -----------------------------
# Create Mask R-CNN model
# -----------------------------
model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)

# -----------------------------
# Train model
# -----------------------------
TRAIN_HERE = True
if TRAIN_HERE:
    NUM_EPOCHS = 300
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=NUM_EPOCHS,
                layers='all')

dir_names=[]
dir_names = next(os.walk(MODEL_DIR))[1]

key = config.NAME.lower()
dir_names = filter(lambda f: f.startswith(key), dir_names)
dir_names = sorted(dir_names)
print(key,dir_names)

if not dir_names:
    import errno
    raise FileNotFoundError(
        errno.ENOENT,
        "Could not find model directory under {}".format(MODEL_DIR))

for d in dir_names:
    dir_name = os.path.join(MODEL_DIR, d)
    os.chown(dir_name, HOST_UID, HOST_GID)
    os.chmod(dir_name, 0o777)

    for root, dirs, files in os.walk(dir_name):
        # Change ownership and permissions for directories
        for d1 in dirs:
            dir_path = os.path.join(root, d1)
            os.chown(dir_path, HOST_UID, HOST_GID)
            os.chmod(dir_path, 0o777)  # rwxrwxrwx for directories
        # Change ownership and permissions for files
        for f in files:
            file_path = os.path.join(root, f)
            os.chown(file_path, HOST_UID, HOST_GID)
            os.chmod(file_path, 0o666)  # rw-rw-rw- for files

fps = []
# Pick last directory
for d in dir_names:
    dir_name = os.path.join(MODEL_DIR, d)
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        print('No weight files in {}'.format(dir_name))
    else:
      checkpoint = os.path.join(dir_name, checkpoints[-1])
      fps.append(checkpoint)

model_path = sorted(fps)[-1]
print('Found model {}'.format(model_path))

# -----------------------------
# Inference config
# -----------------------------
class InferenceConfig(DetectorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

model_infer = modellib.MaskRCNN(mode='inference',
                                config=inference_config,
                                model_dir=MODEL_DIR)

# Load pre-trained weights (replace with your path)
#model_path = os.path.join(MODEL_DIR, 'pneumonia20251222T1623/mask_rcnn_pneumonia_0002.weights.h5')

model_infer.load_weights(model_path, by_name=True)


# -----------------------------
# Inference & visualization
# -----------------------------
def get_colors_for_class_ids(class_ids):
    return [(.941, .204, .204) if cid == 1 else (0, 0, 0) for cid in class_ids]


dataset = dataset_val
#fig = plt.figure(figsize=(10, 30))
for i in range(30):
    image_id = random.choice(dataset.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, inference_config,
                                                                                       image_id)

    #plt.subplot(6, 2, 2 * i + 1)
    # ---- Create figure with 1 row, 2 columns ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                                dataset.class_names,
                                colors=get_colors_for_class_ids(gt_class_id),title='GT',figAx=(fig, axes[0]))

    #plt.subplot(6, 2, 2 * i + 2)
    results = model_infer.detect([original_image])
    r = results[0]

    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset.class_names, r['scores'],
                                colors=get_colors_for_class_ids(r['class_ids']),title='pred',figAx=(fig, axes[1]))

    # ---- Save & close ----
    plt.tight_layout()
    fig_path=DATA_DIR+'/results/'+str(image_id)+'.png'
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    # Change ownership to host user
    os.chown(fig_path, HOST_UID, HOST_GID)
    # Optional: make sure read/write permissions are set
    os.chmod(fig_path, 0o666)  # rw-rw-rw-

    plt.show()
    plt.close(fig)


# -----------------------------
# Predict on test dataset
# -----------------------------
test_image_fps = get_dicom_fps(test_dicom_dir)

filepath_submission=DATA_DIR+'/results/'+'sample_submission.csv'
def predict(image_fps, filepath=filepath_submission, min_conf=0.98):
    with open(filepath_submission, 'w') as file:
        for image_id in tqdm(image_fps):
            ds = pydicom.dcmread(image_id)
            image = ds.pixel_array
            if len(image.shape) != 3 or image.shape[2] != 3:
                image = np.stack((image,) * 3, -1)
            patient_id = os.path.splitext(os.path.basename(image_id))[0]
            results = model_infer.detect([image])
            r = results[0]
            out_str = patient_id
            if len(r['rois']) > 0:
                out_str += ","
                for i in range(len(r['rois'])):
                    if r['scores'][i] > min_conf:
                        x1 = r['rois'][i][1]
                        y1 = r['rois'][i][0]
                        width = r['rois'][i][3] - x1
                        height = r['rois'][i][2] - y1
                        out_str += f" {round(r['scores'][i], 2)} {x1} {y1} {width} {height}"
            file.write(out_str + "\n")


# Example: predict first 50 test images
predict(test_image_fps[:50], filepath=filepath_submission)
os.chown(filepath_submission, HOST_UID, HOST_GID)
# Optional: make sure read/write permissions are set
os.chmod(filepath_submission, 0o666)  # rw-rw-rw-
output = pd.read_csv(filepath_submission, names=['id', 'pred_string'])
output.head()

