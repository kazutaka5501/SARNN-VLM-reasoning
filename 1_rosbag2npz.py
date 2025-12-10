# -*- coding: utf-8 -*-
# Usage: python3 1_rosbag2npz.py /path/to/bag_dir

import os
import cv2
import glob
import rosbag
import argparse
import numpy as np
from cv_bridge import CvBridge

parser = argparse.ArgumentParser()
parser.add_argument('bag_dir', type=str, help='Directory containing .bag files')
args = parser.parse_args()

bag_files = sorted(glob.glob(os.path.join(args.bag_dir, '*.bag')))
bridge = CvBridge()


def process_bag(file):
    print(f"[INFO] Processing: {os.path.basename(file)}")
    savename = file.replace('.bag', '.npz')

    try:
        bag = rosbag.Bag(file)

        # Only include topics that actually exist in your bags
        REQUIRED_TOPICS = [
            '/camera/color/image_raw',
            '/camera/depth/image_rect_raw',
            '/gelsight1/raspicam_node/image/compressed',
            '/gelsight2/raspicam_node/image/compressed',
            '/joint_states',
        ]

        # Buffers for each modality
        image_color = []
        image_depth = []
        image_gelsight1 = []
        image_gelsight2 = []
        joint_state_list = []

        cam_K = None
        cam_D = None
        cam_size = None

        for topic, msg, _ in bag.read_messages(topics=REQUIRED_TOPICS):

            if topic == '/camera/color/image_raw':
                try:
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    image_color.append(img.astype(np.uint8))
                except Exception as e:
                    print(f"[ERROR] Color image decode error: {e}")

            elif topic == '/camera/depth/image_rect_raw':
                try:
                    depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                    if depth.dtype != np.uint16:
                        depth = depth.astype(np.uint16)
                    image_depth.append(depth)
                except Exception as e:
                    print(f"[ERROR] Depth image decode error: {e}")

            elif topic == '/gelsight1/raspicam_node/image/compressed':
                try:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    image_gelsight1.append(im.astype(np.uint8))
                except Exception as e:
                    print(f"[ERROR] Gelsight1 decode error: {e}")

            elif topic == '/gelsight2/raspicam_node/image/compressed':
                try:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    image_gelsight2.append(im.astype(np.uint8))
                except Exception as e:
                    print(f"[ERROR] Gelsight2 decode error: {e}")

            elif topic == '/joint_states':
                # Use only joint positions as state
                joint_state_list.append(np.array(msg.position, dtype=np.float32))

        bag.close()

        # Require at least 2 frames to define (state, next_state) pairs
        if len(image_color) < 2 or len(joint_state_list) < 2:
            print(f"[WARN] Not enough data in {file} "
                  f"(color={len(image_color)}, joint_states={len(joint_state_list)})")
            return

        # Align sequence length, then T = M - 1 for (state, action)
        M = min(len(image_color), len(joint_state_list))
        T = M - 1

        # State: 0 ... T-1
        color_arr = np.array(image_color[:T], dtype=np.uint8)
        joints_state_arr = np.array(joint_state_list[:T], dtype=np.float32)

        # Action: 1 ... M-1
        joints_action_arr = np.array(joint_state_list[1:M], dtype=np.float32)

        save_dict = {
            'color': color_arr,               # (T, H, W, 3)
            'joints_state': joints_state_arr, # (T, P)
            'actions': joints_action_arr,     # (T, P)
        }

        if len(image_depth) >= M:
            save_dict['depth'] = np.array(image_depth[:T], dtype=np.uint16)

        if len(image_gelsight1) >= M:
            save_dict['gelsight1'] = np.array(image_gelsight1[:T], dtype=np.uint8)

        if len(image_gelsight2) >= M:
            save_dict['gelsight2'] = np.array(image_gelsight2[:T], dtype=np.uint8)

        if cam_K is not None:
            save_dict['cam_K'] = cam_K
            save_dict['cam_D'] = cam_D
            save_dict['cam_size'] = cam_size

        np.savez_compressed(savename, **save_dict)

        print(
            f"[OK] Saved: {os.path.basename(savename)} "
            f"(T={T}, color={len(image_color)}, "
            f"depth={'yes' if len(image_depth) else 'no'}, "
            f"gelsight1={'yes' if len(image_gelsight1) else 'no'}, "
            f"gelsight2={'yes' if len(image_gelsight2) else 'no'})"
        )

    except Exception as e:
        print(f"[FATAL] Failed to process {file}: {e}")


if __name__ == '__main__':
    # Process each bag sequentially (more stable, easier to debug)
    for f in bag_files:
        process_bag(f)


