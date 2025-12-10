CLIP-SARNN for Multimodal Cable Manipulation with Vision + Tactile (GelSight)

This repository provides a full multimodal robotic manipulation pipeline using RGB, depth, and tactile GelSight images, together with a CLIP-conditioned Sequential Attentive Recurrent Neural Network (SARNN).
The system extends the methodology of EIPL (Embodied Intelligence with Deep Predictive Learning)—developed by Ogata Laboratory, Waseda University—and integrates modern vision-language models for task-conditioned behavior generation.

Background: About EIPL (Ogata Lab, Waseda University)

This project builds on top of EIPL, an open-source robotics learning framework:

EIPL GitHub: https://github.com/ogata-lab/eipl

EIPL Docs: https://ogata-lab.github.io/eipl-docs/en/

EIPL provides:

Predictive learning models for robot control

Data handling tools (e.g., variable-length sequence padding)

Standardized training utilities

Reproducible motion generation pipelines

This repository follows EIPL’s conventions for dataset formatting, sequence prediction tasks, and network training.
We extend EIPL with:

GelSight tactile modalities

CLIP-guided spatial attention

Text-conditioned task behavior

Multimodal sequence learning via SARNN

Key Features of This Repository
Multimodal Inputs

RGB images

Depth images

GelSight tactile images (left/right tactile cameras)

Robot joint states

Natural language task instructions via CLIP

CLIP-Guided SARNN

Extracts spatial keypoints from high-level CLIP visual embeddings

Learns temporal dynamics via LSTM

Predicts:

next image

next joint state

next keypoint distribution

✔ End-to-End Pipeline from ROSBag → Training

Convert ROSBags → NPZ (1_rosbag2npz.py)

Build padded multimodal dataset (2_make_dataset.py)

Train the CLIP-SARNN model (train_clip_sarnn.py)

Repository Structure
.
├── 1_rosbag2npz.py         # Convert ROS bag → multi-modal npz
├── 2_make_dataset.py       # Build train/test dataset + task labels
├── SARNN_CLIP.py           # CLIP + SpatialSoftmax + SARNN model
├── train_clip_sarnn.py     # Training script
│
├── data/
│   ├── train/
│   ├── test/
│   ├── task_dict.json      # Natural language task instructions
│   └── joint_bounds.npy    # For normalization
│
└── README.md               # This file

1. ROSBag → NPZ Conversion

Run:

python3 1_rosbag2npz.py /path/to/bagfiles

Extracted modalities include:
Topic	Meaning
/camera/color/image_raw	RGB frames
/camera/depth/image_rect_raw	Depth (uint16)
/gelsight1/.../image/compressed	Left GelSight tactile image
/gelsight2/.../image/compressed	Right GelSight tactile image
/joint_states	Robot joint positions
GelSight Information

gelsight1 and gelsight2 are tactile image sensors providing:

high-resolution surface deformation

contact geometry

slip cues

force distribution patterns

They are processed exactly like camera RGB frames.

2. Dataset Generation

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


Tasks are mapped to CLIP text embeddings during training.

3. Model: CLIP-SARNN

Core components:

✔CLIP Visual Encoder

Extracts 7×7 spatial feature map from ViT-B/32

Text embeddings modulate spatial attention

Adapter CNN reduces 512 → K keypoint maps

Spatial Softmax

Converts feature maps → 2D keypoint coordinates.

SARNN (Sequential Attentive RNN)

Predicts over time:

reconstructed image

next joint angles

predicted keypoints

4. Training

Run:

python3 train_clip_sarnn.py --tag clip_run1


Main arguments:

Argument	Description
--no_clip	Train without CLIP (use CNN keypoints only)
--k_dim	Number of learned keypoints
--rec_dim	LSTM hidden size
--stdev	Image noise augmentation
--epoch	Number of training steps

Logs are stored in TensorBoard:

tensorboard --logdir log/

 Training Output

The script saves:

Best model checkpoint: CLIP_SARNN.pth

Loss curves: loss.png

TensorBoard logs

NPZ File Structure

Each .npz file contains:

{
    "color":       (T, H, W, 3),
    "depth":       (T, H, W),
    "gelsight1":   (T, H, W, 3),
    "gelsight2":   (T, H, W, 3),
    "joints_state": (T, P),
    "actions":      (T, P)
}
