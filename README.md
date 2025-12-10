## CLIP-SARNN Multimodal Manipulation Package (Vision + GelSight + Language)

This repository provides a multimodal sequence-prediction and manipulation learning pipeline built upon RGB vision, depth sensing, GelSight tactile images, and natural-language task instructions.
It is designed as a research-oriented extension to predictive sensorimotor learning frameworks such as EIPL (Embodied Intelligence with Deep Predictive Learning).

This package does not replace any ROS driver or hardware API. Instead, it provides the data processing, model architecture, training pipeline, and CLIP-based task-conditioning suitable for academic research in multimodal robotic manipulation.

## Features

 Multimodal input support (RGB / Depth / GelSight tactile images / Joint States)

 CLIP-guided spatial keypoint extraction

 Natural-language task conditioning

 SARNN temporal dynamics model

 ROSBag → NPZ conversion pipeline

 Dataset builder with task labels

## Component Overview
Component	File	Description
ROSBag → NPZ Converter	1_rosbag2npz.py	Extracts RGB, depth, GelSight1/2 tactile images, joint states, and generates action sequences.
Dataset Builder	2_make_dataset.py	Creates padded train/test datasets with task labels and normalization metadata.
CLIP-SARNN Model	SARNN_CLIP.py	Implements CLIP-guided keypoint extraction, Spatial Softmax, and SARNN temporal prediction.
Training Pipeline	train_clip_sarnn.py	Full training loop with text embedding lookup, mixed precision, scheduling, and logging.
Requirements
Item	Version
Ubuntu	20.04+
Python	3.8+
PyTorch	1.11+ (CUDA recommended)
CLIP	openai-clip
ROS (optional, for data collection)	Noetic / ROS2
GelSight modules	Required if using tactile input
EIPL	Optional but recommended

**1. ROSBag → NPZ Dataset Conversion**

Run the converter on a directory of ROS bag files:

python3 1_rosbag2npz.py /path/to/bag_dir

Extracted modalities
Topic	Description
/camera/color/image_raw	RGB frames
/camera/depth/image_rect_raw	Depth images (uint16)
/gelsight1/.../image/compressed	Left GelSight tactile image
/gelsight2/.../image/compressed	Right GelSight tactile image
/joint_states	Robot joint angles
Output structure inside each .npz:
color         (T, H, W, 3)
depth         (T, H, W)
gelsight1     (T, H, W, 3)
gelsight2     (T, H, W, 3)
joints_state  (T, P)
actions       (T, P)    # q(t+1)


GelSight cameras supply tactile deformation patterns used as image-based tactile observations.

**2. Dataset Construction**

Run:

python3 2_make_dataset.py


Outputs:

data/train/images.npy
data/train/joints.npy
data/train/tasks.npy
data/test/images.npy
data/test/joints.npy
data/test/tasks.npy
data/task_dict.json
data/joint_bounds.npy

Example task dictionary:
{
    "0": "grasp the black cable",
    "1": "pull the cable to the left",
    "2": "push the cable forward"
}


These strings are encoded into CLIP text embeddings and used for task conditioning.

**3. CLIP-SARNN Model**

This model integrates three major components:

(1) CLIP Visual Encoder

Uses ViT-B/32

Extracts a 7×7 spatial feature map

Text embedding modulates channel attention

Adapter CNN reduces 512 → K heatmap channels

(2) Spatial Softmax

Converts each heatmap into 2D keypoints.

(3) SARNN Temporal Dynamics

LSTMCell-based recursive prediction

Predicts:

next-frame image reconstruction

next joint state

next keypoint distribution

The architecture is designed for long-horizon multimodal prediction.

**4. Training the Model**

Run:

python3 train_clip_sarnn.py --tag clip_run1

Key arguments
Argument	Function
--no_clip	Disable CLIP encoder
--k_dim	Number of keypoints
--rec_dim	Size of RNN hidden state
--stdev	Image noise augmentation
--batch_size	Training batch size
--epoch	Number of epochs
Training logs:
tensorboard --logdir log/


Outputs:

CLIP_SARNN.pth
loss.png
TensorBoard event files

**5. NPZ Dataset Format**

Each .npz file contains:

{
    "color": (T, H, W, 3),
    "depth": (T, H, W),
    "gelsight1": (T, H, W, 3),
    "gelsight2": (T, H, W, 3),
    "joints_state": (T, P),
    "actions": (T, P)
}


This structure aligns with predictive-learning conventions used in EIPL.

**6. Relation to EIPL**

This repository is heavily inspired by the concepts and architecture of:

EIPL: Embodied Intelligence with Deep Predictive Learning
https://github.com/ogata-lab/eipl


https://ogata-lab.github.io/eipl-docs/en/

EIPL provides the theoretical and software foundations for:

future-state prediction

attention-based representation learning

recurrent predictive control

Our contribution expands EIPL’s paradigm by integrating:

CLIP language conditioning

GelSight tactile observation

multimodal fusion for manipulation tasks
