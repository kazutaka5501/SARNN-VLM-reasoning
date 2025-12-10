# -*- coding: utf-8 -*-
# Usage: python3 2_make_dataset.py

import os
import glob
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from eipl.utils import list_to_numpy  # ensure eipl is installed


# Configure paths (modify according to your environment)
DATA_ROOT = "/media/joy/SSD2TB/20251119/one1"       # directory containing .npz files
SAVE_ROOT = "/home/joy/work/tmp/2025gelsight/1024/data"  # output directory

# Image configuration
TARGET_SIZE = (192, 192)  # SARNN input size
PREFERRED_IMAGE_KEY = 'color'  # 'color', 'gelsight1', etc.


def calc_minmax(data):
    # Compute min and max values for normalization reference
    return np.min(data, axis=(0, 1)), np.max(data, axis=(0, 1))


def _resize_seq(x, target_h, target_w):
    # Resize a sequence of images to target size
    # x: (T, H, W, C)
    T, H, W, C = x.shape

    if H == target_h and W == target_w:
        return x

    out = np.empty((T, target_h, target_w, C), dtype=x.dtype)

    for t in range(T):
        frame = x[t]
        if C == 1:
            frame2d = frame[..., 0]
            resized = cv2.resize(frame2d, (target_w, target_h), interpolation=cv2.INTER_AREA)
            out[t, ..., 0] = resized
        else:
            out[t] = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return out


def load_data(data_dir):
    # Load all .npz files and extract image and joint sequences
    all_images = []
    all_joints = []
    seq_lengths = []

    npz_files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    print(f"[INFO] Found {len(npz_files)} npz files.")

    for path in npz_files:
        try:
            data = np.load(path, allow_pickle=False)

            # 1. Load image (support multimodal concatenation if needed)
            # Current logic: only color is used; modify to include depth/gelsight if desired
            if 'color' not in data:
                print(f"[WARN] No color image in {path}, skipping.")
                continue

            color = data['color']  # (T, H, W, 3)
            color = _resize_seq(color, TARGET_SIZE[0], TARGET_SIZE[1])

            images = color  # extend here if adding depth/gelsight channels

            # 2. Load joint states
            if 'joints_state' not in data:
                continue

            joints = data['joints_state'].astype(np.float32)

            # 3. Align sequence length
            T = min(len(images), len(joints))
            images = images[:T]
            joints = joints[:T]

            all_images.append(images)
            all_joints.append(joints)
            seq_lengths.append(T)

        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue

    if not all_images:
        raise RuntimeError("No valid data loaded.")

    # Convert list of variable-length sequences into numpy array with padding
    max_len = max(seq_lengths)
    images_np = list_to_numpy(all_images, max_len)
    joints_np = list_to_numpy(all_joints, max_len)

    return images_np, joints_np


def main():
    os.makedirs(os.path.join(SAVE_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_ROOT, "test"), exist_ok=True)

    # 1. Load dataset
    print("Loading data...")
    images, joints = load_data(DATA_ROOT)
    N = images.shape[0]
    print(f"[INFO] Total sequences: {N}, Image shape: {images.shape[1:]}, Joint shape: {joints.shape[1:]}")

    # 2. Task instructions (you can modify these)
    task_instructions_map = {
        0: "grasp the black cable",
        1: "pull the cable to the left",
        2: "push the cable forward"
    }

    # 3. Generate task labels
    tasks = np.zeros(N, dtype=np.int32)

    # Assign tasks: first third → task0, middle third → task1, last third → task2
    part_size = N // 3
    tasks[0:part_size] = 0
    tasks[part_size: part_size*2] = 1
    tasks[part_size*2:] = 2

    print(f"[INFO] Task distribution: Task0={np.sum(tasks==0)}, Task1={np.sum(tasks==1)}, Task2={np.sum(tasks==2)}")

    # 4. Train/test split (stratified by task)
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=tasks
    )

    train_idx.sort()
    test_idx.sort()

    print(f"[INFO] Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # 5. Save dataset
    print("Saving datasets...")

    # Train set
    np.save(os.path.join(SAVE_ROOT, "train", "images.npy"), images[train_idx].astype(np.uint8))
    np.save(os.path.join(SAVE_ROOT, "train", "joints.npy"), joints[train_idx].astype(np.float32))
    np.save(os.path.join(SAVE_ROOT, "train", "tasks.npy"), tasks[train_idx])

    # Test set
    np.save(os.path.join(SAVE_ROOT, "test", "images.npy"), images[test_idx].astype(np.uint8))
    np.save(os.path.join(SAVE_ROOT, "test", "joints.npy"), joints[test_idx].astype(np.float32))
    np.save(os.path.join(SAVE_ROOT, "test", "tasks.npy"), tasks[test_idx])

    # Save task dictionary
    with open(os.path.join(SAVE_ROOT, "task_dict.json"), "w") as f:
        json.dump(task_instructions_map, f, indent=4)

    # Save joint min/max for normalization
    joint_bounds = calc_minmax(joints)
    np.save(os.path.join(SAVE_ROOT, "joint_bounds.npy"), joint_bounds)

    print("[SUCCESS] Dataset generation complete.")
    print(f"Data saved to: {SAVE_ROOT}")
    print("Generated: images.npy, joints.npy, tasks.npy, task_dict.json")


if __name__ == "__main__":
    main()

