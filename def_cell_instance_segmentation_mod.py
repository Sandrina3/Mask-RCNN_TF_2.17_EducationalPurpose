''' ON DOCKER TERMINAL
pip install scikit-image matplotlib imgaug
apt-get update && apt-get install -y libgl1 libglib2.0-0
'''

'''
!pip3 install cython==3.0.5
!pip3 install h5py==3.9.0
!pip3 install imgaug==0.4.0
!pip3 install ipython==7.34.0
!pip3 install ipython-genutils==0.2.0
!pip3 install ipython-sql==0.5.0
!pip3 install keras==2.14.0
!pip3 install matplotlib==3.7.1
!pip3 install numpy==1.23.5
!pip3 install opencv-contrib-python==4.8.0.76
!pip3 install opencv-python==4.8.0.76
!pip3 install pillow==9.4.0
#!pip3 install scikit-image==0.19.3
#!pip3 install scipy==1.11.3
#!pip3 install tensorboard==2.14.1
!pip3 install tensorflow==2.12.0
!pip3 install pydicom==2.4
#!pip3 install -q tqdm
'''

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

#import logging
#logger = tf.get_logger()
#logger.setLevel(logging.ERROR) # or logging.INFO, logging.WARNING, etc.

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(tf.config.list_physical_devices('GPU'))
else:
    print("No GPU device found")

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

print(tf.executing_eagerly())
input()

from mrcnn2 import utils
from mrcnn2 import visualize
from mrcnn.visualize import display_images
from mrcnn2 import model as modellib
from mrcnn2.model import log
from mrcnn2 import cell
from mrcnn import config

# Commented out IPython magic to ensure Python compatibility.
import os
import sys
import itertools
import math
import logging
import json
import re
import random
import time
import concurrent.futures
import numpy as np
print(np.__version__)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
#import imgaug
if not hasattr(np, "bool"):
    np.bool = bool
from imgaug import augmenters as iaa
import pandas as pd

HOST_UID = 1000
HOST_GID = 1000

# Root directory of the project
import os
ROOT_DIR = '/workspace/data/sartorius-cell-instance-segmentation'
DATASET_DIR = os.path.join(ROOT_DIR, "cell")

#config=config.Config()
config = cell.CellConfig()
config.display()

# Load dataset
dataset = cell.CellDataset() # call the CellDataset class from the cell.py package

dataset.load_cell(DATASET_DIR, subset="train")

# Must call before using the dataset
dataset.prepare() # prepares the dataset built-in function

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
  print("{:3}. {:50}".format(i, info['name']))

"""## Display samples"""

image_ids = np.random.choice(dataset.image_ids, 10)

for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    #print(mask,mask.shape, class_ids)
    fig=plt.figure(figsize=(18,9))
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)
    # ---- Save & close ----
    plt.tight_layout()
    fig_path=DATASET_DIR + '/' +str(image_id)+'.png'
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    # Change ownership to host user
    os.chown(fig_path, HOST_UID, HOST_GID)
    # Optional: make sure read/write permissions are set
    os.chmod(fig_path, 0o666)  # rw-rw-rw-

    plt.show()
    plt.close()
## se usi config=config.Config() devi modificare "crop" e "cell"
#config.IMAGE_RESIZE_MODE="crop"
#config.NAME="cell"

## se usi config = cell.CellConfig() devi modificare False mini mask
config.USE_MINI_MASK=True

config.display()
# overlaying the original image & mask together
# Example of loading a specific image by its source ID
source_id = "053d61766edb"

# Map source ID to Dataset image_id
# Notice the nucleus prefix: it's the name given to the dataset in NucleusDataset
image_id = dataset.image_from_source_map["cell.{}".format(source_id)]

# Load and display
image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
        dataset, config, image_id)
log("molded_image", image)
log("mask", mask)

# Now call visualize.display_instances with the resized mask
# Suppose image shape is (256, 256)

full_masks = np.zeros(
    (image.shape[0], image.shape[1], mask.shape[2]),
    dtype=np.uint8
)

for i in range(mask.shape[2]):
    mini_mask = mask[:, :, i].astype(np.uint8)  # <<< IMPORTANT
    full_masks[:, :, i] = utils.unmold_mask(
        mini_mask, bbox[i], image.shape[:2]
    )

fig, ax = plt.subplots(1, 1, figsize=(18, 9))
visualize.display_instances(image, bbox, full_masks, class_ids, dataset.class_names,
                            show_bbox=False, show_mask=True,figAx=(fig, ax) )

## ---- Save & close ----
plt.tight_layout()
fig_path = DATASET_DIR + '/prova.png'
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
# Change ownership to host user
os.chown(fig_path, HOST_UID, HOST_GID)
# Optional: make sure read/write permissions are set
os.chmod(fig_path, 0o666)  # rw-rw-rw-

plt.show()
plt.close()


"""# Model ## Initialization"""

# Data path
TRAIN_PATH = 'cell/train/'
TEST_PATH ='cell/test'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from release if needed
if not os.path.exists(COCO_MODEL_PATH):
  utils.download_trained_weights(COCO_MODEL_PATH)
  os.chown(COCO_MODEL_PATH, HOST_UID, HOST_GID)
  os.chmod(COCO_MODEL_PATH, 0o666)  # rw-rw-rw-

"""## Dataset loader"""

import shutil

DATASET_DIR = os.path.join(ROOT_DIR, "cell")

source_dir = os.path.join(DATASET_DIR, 'train')
# target_dir = os.path.join(DATASET_DIR, 'val_set')
#
# file_names = os.listdir(source_dir)[:25]
#
# for file_name in file_names:
#   shutil.copy(os.path.join(source_dir, file_name), target_dir)

# Training dataset
dataset_train = cell.CellDataset()
dataset_train.load_cell(DATASET_DIR, subset="train")
dataset_train.prepare()

# Validation dataset
dataset_val = cell.CellDataset()
dataset_val.load_cell(DATASET_DIR, subset="val")
dataset_val.prepare()

# # -----------------------------
# # Visualize images
# # -----------------------------
#
# # Choose dataset
# dataset_chosen = dataset_val  # train set
#
# # Display 4 random images from train set
# for i in range(10):
#     image_id = random.choice(dataset_chosen.image_ids)
#     # Load image and ground truth data
#     # Load and display
#     image, image_meta, class_ids, bbox, mask = modellib.load_image_gt(
#         dataset_chosen, config, image_id)
#     log("molded_image", image)
#     log("mask", mask)
#
#     # Now call visualize.display_instances with the resized mask
#     # Suppose image shape is (256, 256)
#
#     full_masks = np.zeros(
#         (image.shape[0], image.shape[1], mask.shape[2]),
#         dtype=np.uint8
#     )
#
#     for i in range(mask.shape[2]):
#         mini_mask = mask[:, :, i].astype(np.uint8)  # <<< IMPORTANT
#         full_masks[:, :, i] = utils.unmold_mask(
#             mini_mask, bbox[i], image.shape[:2]
#         )
#
#     fig, ax = plt.subplots(1, 1, figsize=(18, 9))
#     visualize.display_instances(image, bbox, full_masks, class_ids, dataset_chosen.class_names,
#                                 show_bbox=False, show_mask=True, figAx=(fig, ax))
#
#     # ---- Save & close ----
#     plt.tight_layout()
#     fig_path=DATASET_DIR+'/'+str(image_id)+'.png'
#     plt.savefig(fig_path, dpi=200, bbox_inches="tight")
#     # Change ownership to host user
#     os.chown(fig_path, HOST_UID, HOST_GID)
#     # Optional: make sure read/write permissions are set
#     os.chmod(fig_path, 0o666)  # rw-rw-rw-
#
#     plt.show()
#     plt.close(fig)

"""## Create Model"""

# Create Model
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

TRAIN_HERE=True

# Initialize coco weight
if TRAIN_HERE==True:

  init_with = "coco"

  model.load_weights(COCO_MODEL_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                              "mrcnn_bbox", "mrcnn_mask"])

"""# Data Augmentation"""

augmentation = iaa.SomeOf((0,2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270)
        ]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
])

"""# Training

Train in two stages:

1.  Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized pretrained weights from MS COCO). To train only the head layers, pas `layers='heads'` to the `train()` function.

2. Fine-tune all layers.

"""

if TRAIN_HERE==True:

  # Train the head
  model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE,
              epochs=2,
              #augmentation=augmentation,
              layers='heads')

if TRAIN_HERE==True:

  # Fine tune all layers
  model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE / 10,
              epochs=2,
              #augmentation=augmentation,
              layers='all')

"""# Inference"""

import glob
import skimage
import imageio

inference_config = cell.CellInferenceConfig()
inference_config.display()

# Recreate the model in inference mode
model_infer = modellib.MaskRCNN(mode="inference", config=inference_config,
                                model_dir=MODEL_DIR)

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

# Load trained weights
model_infer.load_weights(model_path, by_name=True)

train = pd.read_csv(ROOT_DIR + '/cell/train.csv')

# Unique Image IDs
id_unique = train['id'].unique()

# Original Image File Path
def get_file_path(image_id):
    return f'/{ROOT_DIR}/cell/train/{image_id}.png'

train['file_path'] = train['id'].apply(get_file_path)

# Unique Cell Names
CELL_NAMES = np.sort(train['cell_type'].unique())
print(f'CELL_NAMES: {CELL_NAMES}')

for file_path in glob.glob(ROOT_DIR + '/cell/train/*.png')[:5]:
    img = skimage.io.imread(file_path)
    img = np.expand_dims(img, axis=2)
    img = np.concatenate((img, img, img), axis=2)
    results = model_infer.detect([img], verbose=1)
    r = results[0]

    # Image Id
    image_id = file_path.split('/')[-1].split('.')[0]
    print(f'image_id: {image_id}')

    mask = cell.rle_decode(image_id, ROOT_DIR + '/cell/train.csv' )
    mask = np.sum(mask, axis=2)
    plt.figure(figsize=(16,16))
    plt.imshow(mask)
    #plt.show()

    visualize.display_instances(
        img,
        r['rois'],
        r['masks'],
        r['class_ids'],
        ['BG'] + CELL_NAMES.tolist(),
        r['scores'],
        figsize=(16,16)
    )

    # ---- Save & close ----
    plt.tight_layout()
    fig_path=DATASET_DIR+'/results/'+str(image_id)+'.png'
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    # Change ownership to host user
    os.chown(fig_path, HOST_UID, HOST_GID)
    # Optional: make sure read/write permissions are set
    os.chmod(fig_path, 0o666)  # rw-rw-rw-

    plt.show()
    plt.close()