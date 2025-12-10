# 2025 SARNN + CLIP Training Script (Clean Version)
# This version removes docstrings, Chinese text, emojis, and decorative lines.

import os
import sys
import json
import argparse
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from eipl.utils import EarlyStopping, check_args, tensor2numpy, LossScheduler


# Dataset ---------------------------------------------------------
class CLIPMultimodalDataset(torch.utils.data.Dataset):
    # Dataset now includes task indices for text embedding lookup
    def __init__(self, images, joints, tasks, device=None, stdev=None):
        super().__init__()
        self.images = images
        self.joints = joints
        self.tasks = tasks
        self.device = device
        self.stdev = stdev

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        y_img = self.images[idx]
        y_joint = self.joints[idx]
        task_idx = self.tasks[idx]

        if self.stdev is not None and self.stdev > 0:
            noise = torch.normal(mean=0.0, std=self.stdev, size=y_img.shape, device=y_img.device)
            x_img = y_img + noise
        else:
            x_img = y_img

        x_joint = y_joint
        return (x_img, x_joint, task_idx), (y_img, y_joint)


# Model Import ---------------------------------------------------------
sys.path.append("/home/joy/work/tmp/2025gelsight/1024")
from SARNN_CLIP import SARNN


# Trainer ---------------------------------------------------------
class CLIP_BPTT_Trainer:
    def __init__(self, model, optimizer, text_emb_table, loss_weights=[1.0, 1.0, 1.0], device="cpu"):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)
        self.scaler = GradScaler()
        self.text_emb_table = text_emb_table.to(self.device) if text_emb_table is not None else None

    def save(self, epoch, loss, savename):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "train_loss": loss[0],
            "test_loss": loss[1],
        }, savename)

    def process_epoch(self, data_loader, training=True):
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        n_batch = 0

        for n_batch, ((x_img, x_joint, task_indices), (y_img, y_joint)) in enumerate(data_loader):
            x_img = x_img.to(self.device)
            y_img = y_img.to(self.device)
            x_joint = x_joint.to(self.device)
            y_joint = y_joint.to(self.device)
            task_indices = task_indices.to(self.device)

            batch_text = None
            if self.text_emb_table is not None:
                batch_text = self.text_emb_table[task_indices]

            state = None
            yi_list, yv_list = [], []
            dec_pts_list, enc_pts_list = []

            self.optimizer.zero_grad(set_to_none=True)

            with autocast():
                for t in range(x_img.shape[1] - 1):
                    yi_hat_t, yv_hat_t, enc_t, dec_t, state = self.model(
                        xi=x_img[:, t],
                        xv=x_joint[:, t],
                        state=state,
                        text_embedding=batch_text
                    )
                    yi_list.append(yi_hat_t)
                    yv_list.append(yv_hat_t)
                    enc_pts_list.append(enc_t)
                    dec_pts_list.append(dec_t)

                yi_hat = torch.permute(torch.stack(yi_list), (1, 0, 2, 3, 4))
                yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))

                img_loss = nn.MSELoss()(yi_hat, y_img[:, 1:]) * self.loss_weights[0]
                joint_loss = nn.MSELoss()(yv_hat, y_joint[:, 1:]) * self.loss_weights[1]
                pt_loss = nn.MSELoss()(
                    torch.stack(dec_pts_list[:-1]), torch.stack(enc_pts_list[1:])
                ) * self.scheduler(self.loss_weights[2])

                loss = img_loss + joint_loss + pt_loss

            total_loss += tensor2numpy(loss)

            if training:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

        if n_batch == 0:
            return total_loss
        return total_loss / (n_batch + 1)


# Main ---------------------------------------------------------
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=6000)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--rec_dim", type=int, default=128)
parser.add_argument("--k_dim", type=int, default=5)
parser.add_argument("--img_loss", type=float, default=1.0)
parser.add_argument("--joint_loss", type=float, default=1.1)
parser.add_argument("--pt_loss", type=float, default=1.0)
parser.add_argument("--heatmap_size", type=float, default=0.1)
parser.add_argument("--temperature", type=float, default=5e-5)
parser.add_argument("--stdev", type=float, default=0.5)
parser.add_argument("--log_dir", default="log/")
parser.add_argument("--vmin", type=float, default=0.0)
parser.add_argument("--vmax", type=float, default=1.0)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--tag", default="clip_sarnn_test")
parser.add_argument("--no_clip", action="store_true")
args = parser.parse_args()
args = check_args(args)

DATA_ROOT = "/home/joy/work/tmp/2025gelsight/1024/data"
stdev = args.stdev * (args.vmax - args.vmin)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

print(f"Device = {device}, CLIP Enabled = {not args.no_clip}")


# Data Loading ---------------------------------------------------------
def load_npy(name):
    path = os.path.join(DATA_ROOT, name)
    try:
        return np.load(path)
    except FileNotFoundError:
        print(f"[FATAL] Cannot load {path}")
        sys.exit(1)


train_images = load_npy("train/images.npy")
train_joints = load_npy("train/joints.npy")
train_tasks = load_npy("train/tasks.npy")

test_images = load_npy("test/images.npy")
test_joints = load_npy("test/joints.npy")
test_tasks = load_npy("test/tasks.npy")

train_images = torch.tensor(train_images, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0
train_joints = torch.tensor(train_joints, dtype=torch.float32)
train_tasks = torch.tensor(train_tasks, dtype=torch.long)

test_images = torch.tensor(test_images, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0
test_joints = torch.tensor(test_joints, dtype=torch.float32)
test_tasks = torch.tensor(test_tasks, dtype=torch.long)

train_images = train_images[:, ::2]
train_joints = train_joints[:, ::2]
test_images = test_images[:, ::2]
test_joints = test_joints[:, ::2]


train_dataset = CLIPMultimodalDataset(train_images, train_joints, train_tasks, device=device, stdev=stdev)
test_dataset = CLIPMultimodalDataset(test_images, test_joints, test_tasks, device=device, stdev=None)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)


# Model ---------------------------------------------------------
model = SARNN(
    rec_dim=args.rec_dim,
    joint_dim=train_joints.shape[2],
    k_dim=args.k_dim,
    heatmap_size=args.heatmap_size,
    temperature=args.temperature,
    im_size=train_images.shape[-2:],
    in_ch=train_images.shape[2],
    use_clip=(not args.no_clip),
    device=str(device)
).to(device)


# Text Embedding Table ---------------------------------------------------------
text_emb_table = None
if not args.no_clip:
    try:
        with open(os.path.join(DATA_ROOT, "task_dict.json"), "r") as f:
            task_dict_raw = json.load(f)
            max_idx = max([int(k) for k in task_dict_raw.keys()])
            task_list = [task_dict_raw[str(i)] for i in range(max_idx + 1)]

            text_emb_table = model.encode_text(task_list)

    except FileNotFoundError:
        text_emb_table = model.encode_text(["default task"] * 10)


# Training ---------------------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
trainer = CLIP_BPTT_Trainer(
    model, optimizer,
    text_emb_table=text_emb_table,
    loss_weights=[args.img_loss, args.joint_loss, args.pt_loss],
    device=device
)
early_stop = EarlyStopping(patience=20)

log_dir_path = os.path.join(args.log_dir, args.tag)
os.makedirs(log_dir_path, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir_path)
save_name = os.path.join(log_dir_path, "CLIP_SARNN.pth")

train_loss_list, test_loss_list = [], []

with tqdm(range(args.epoch)) as pbar:
    for epoch in pbar:
        torch.cuda.empty_cache()

        train_loss = trainer.process_epoch(train_loader, training=True)
        with torch.no_grad():
            test_loss = trainer.process_epoch(test_loader, training=False)

        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        save_ckpt, _ = early_stop(test_loss)
        if save_ckpt:
            trainer.save(epoch, [train_loss, test_loss], save_name)

        pbar.set_postfix(
            train=f"{train_loss:.4f}",
            test=f"{test_loss:.4f}"
        )


# Plot Loss ---------------------------------------------------------
plt.figure()
plt.plot(train_loss_list, label="Train")
plt.plot(test_loss_list, label="Test")
plt.legend()
plt.savefig(os.path.join(log_dir_path, "loss.png"))
print("Training complete.")

