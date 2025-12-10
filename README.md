**CLIP-SARNN for Multimodal Cable Manipulation with Vision + Tactile (GelSight)**

This repository contains a complete multimodal robotic manipulation pipeline that integrates RGB vision, depth sensing, tactile GelSight images, and natural-language task instructions via CLIP, combined with a Sequential Attentive Recurrent Neural Network (SARNN).

The project extends the predictive-learning framework of EIPL (Embodied Intelligence with Deep Predictive Learning), developed by Ogata Laboratory, Waseda University.

**Background: EIPL**

This project builds on and integrates components from EIPL, an open-source robot learning framework:

EIPL GitHub: https://github.com/ogata-lab/eipl

EIPL Documentation: https://ogata-lab.github.io/eipl-docs/en/

EIPL provides:

Motion prediction and generation models

Dataset utilities (list_to_numpy, padding variable-length sequences, etc.)

Training tools such as EarlyStopping, LossScheduler, and evaluation utilities

A unified structure for predictive sensorimotor learning

Our project extends EIPL with:

Multimodal tactile perception (GelSight1 & GelSight2)

CLIP-guided visual keypoint extraction

Text-conditioned manipulation behavior

A SARNN architecture for long-horizon multimodal prediction

**Features**
Multimodal Inputs

RGB camera images

Depth images

GelSight tactile images

gelsight1 and gelsight2 are high-resolution tactile image sensors providing surface deformation, slip cues, and contact geometry

Robot joint states

Task instructions encoded by CLIP text embeddings

Model Architecture

CLIP (ViT-B/32) feature extraction

Adapter CNN → Spatial Softmax → learned keypoints

SARNN temporal dynamics (LSTMCell)

Predicts:

reconstructed next-frame image

future joint states

next keypoint distribution


**1. Convert ROSBag → NPZ**

Run:

python3 1_rosbag2npz.py /path/to/rosbag_dir


Extracted topics:

Topic	Description
/camera/color/image_raw	RGB images
/camera/depth/image_rect_raw	Depth images (uint16)
/gelsight1/.../image/compressed	Left GelSight tactile RGB
/gelsight2/.../image/compressed	Right GelSight tactile RGB
/joint_states	Robot joint positions

GelSight sensors provide tactile observation used as image input.

The script also computes:

joints_state[t] = q_t

actions[t] = q_{t+1} (for behavior cloning)

**2. Build Dataset**

Run:

python3 2_make_dataset.py


Produces:

data/train/images.npy
data/train/joints.npy
data/train/tasks.npy
data/test/images.npy
data/test/joints.npy
data/test/tasks.npy
data/task_dict.json
data/joint_bounds.npy


Example task_dict.json:

{
    "0": "grasp the black cable",
    "1": "pull the cable to the left",
    "2": "push the cable forward"
}


These instructions are converted into CLIP embeddings during training.

**3. Train the CLIP-SARNN Model**

Run training:

python3 train_clip_sarnn.py --tag clip_run1


Useful arguments:

Argument	Description
--no_clip	Disable CLIP (use CNN encoder only)
--k_dim	Number of keypoints
--rec_dim	RNN hidden size
--stdev	Image noise augmentation
--batch_size	Training batch size

TensorBoard:

tensorboard --logdir log/


Outputs:

CLIP_SARNN.pth (best model checkpoint)

loss.png (loss curves)

TensorBoard summaries

**NPZ File Structure**

Every .npz file contains:

{
    "color": (T, H, W, 3),
    "depth": (T, H, W),
    "gelsight1": (T, H, W, 3),
    "gelsight2": (T, H, W, 3),
    "joints_state": (T, P),
    "actions": (T, P)
}
