import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import clip


def create_position_encoding(width: int, height: int, normalized=True, data_format="channels_first"):
    # Create position encoding maps for x/y coordinates.
    if normalized:
        pos_x, pos_y = np.meshgrid(np.linspace(0.0, 1.0, width),
                                   np.linspace(0.0, 1.0, height),
                                   indexing="xy")
    else:
        pos_x, pos_y = np.meshgrid(np.linspace(0, width - 1, width),
                                   np.linspace(0, height - 1, height),
                                   indexing="xy")

    if data_format == "channels_first":
        pos_xy = torch.from_numpy(np.stack([pos_x, pos_y], axis=0)).float()
    else:
        pos_xy = torch.from_numpy(np.stack([pos_x, pos_y], axis=2)).float()

    pos_x = torch.from_numpy(pos_x.reshape(width * height)).float()
    pos_y = torch.from_numpy(pos_y.reshape(width * height)).float()

    return pos_xy, pos_x, pos_y


class SpatialSoftmax(nn.Module):
    # Spatial Softmax converting heatmaps → keypoints.
    def __init__(self, width: int, height: int, temperature=1e-4, normalized=True):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = torch.nn.Parameter(torch.ones(1)) if temperature is None else temperature

        _, pos_x, pos_y = create_position_encoding(width, height, normalized=normalized)
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        B, C, W, H = x.shape

        # Lazy reinit if resolution mismatch
        if (self.width != int(W)) or (self.height != int(H)) or (self.pos_x.numel() != W * H):
            self.width = int(W)
            self.height = int(H)

            grid_x = torch.linspace(0.0, 1.0, self.width, device=x.device)
            grid_y = torch.linspace(0.0, 1.0, self.height, device=x.device)
            pos_xg, pos_yg = torch.meshgrid(grid_x, grid_y, indexing='ij')

            pos_x = pos_xg.reshape(-1).to(dtype=x.dtype)
            pos_y = pos_yg.reshape(-1).to(dtype=x.dtype)

            self.register_buffer("pos_x", pos_x, persistent=False)
            self.register_buffer("pos_y", pos_y, persistent=False)

        temp = self.temperature if torch.is_tensor(self.temperature) \
            else torch.tensor(self.temperature, device=x.device, dtype=x.dtype)

        x_flat = x.contiguous().view(B, C, -1) / temp
        prob = torch.softmax(x_flat, dim=-1)

        ex = torch.sum(prob * self.pos_x.view(1, 1, -1), dim=-1)
        ey = torch.sum(prob * self.pos_y.view(1, 1, -1), dim=-1)
        keys = torch.stack([ex, ey], dim=-1)

        prob_map = prob.view(B, C, self.width, self.height)
        return keys, prob_map


class InverseSpatialSoftmax(nn.Module):
    # Convert keypoints → Gaussian heatmaps.
    def __init__(self, width: int, height: int, heatmap_size=0.1, normalized=True):
        super().__init__()
        self.height = height
        self.width = width
        self.normalized = normalized
        self.heatmap_size = heatmap_size

        pos_xy, _, _ = create_position_encoding(width, height, normalized=normalized)
        self.register_buffer("pos_xy", pos_xy)

    def forward(self, keys):
        squared_distances = torch.sum(
            torch.pow(self.pos_xy[None, None] - keys[:, :, :, None, None], 2.0),
            axis=2
        )
        heatmap = torch.exp(-squared_distances / self.heatmap_size)
        return heatmap


class CLIPGuidedPosEncoder(nn.Module):
    # Position encoder using CLIP visual features.
    def __init__(self, k_dim, temperature, device='cuda'):
        super().__init__()
        self.k_dim = k_dim

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.dtype = self.clip_model.dtype
        self.adapter_conv = nn.Conv2d(512, k_dim, kernel_size=1).float()
        self.ss = SpatialSoftmax(width=7, height=7, temperature=temperature, normalized=True)

        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1))

    def get_clip_feature_map(self, x):
        # Extract 7×7 CLIP feature map.
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = (x - self.mean) / self.std
        x = x.type(self.dtype)

        x = self.clip_model.visual.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        cls = self.clip_model.visual.class_embedding.to(x.dtype)
        cls = cls + torch.zeros(x.shape[0], 1, cls.shape[-1], dtype=x.dtype, device=x.device)

        x = torch.cat([cls, x], dim=1)
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.visual.transformer(x)
        x = x.permute(1, 0, 2)

        x = x[:, 1:, :]
        x = x.permute(0, 2, 1).reshape(-1, 512, 7, 7)
        return x.float()

    def forward(self, image, text_embedding=None):
        with torch.no_grad():
            vis_feat = self.get_clip_feature_map(image)

        if text_embedding is not None:
            weight = text_embedding.view(vis_feat.shape[0], 512, 1, 1)
            vis_feat = vis_feat * weight

        heatmap_low_res = self.adapter_conv(vis_feat)
        keys, prob_map = self.ss(heatmap_low_res)
        return keys, prob_map


class SARNN(nn.Module):
    # Sequential Attention RNN with optional CLIP keypoint encoder.
    def __init__(
        self,
        rec_dim=128,
        k_dim=5,
        joint_dim=12,
        temperature=5e-5,
        heatmap_size=0.1,
        kernel_size=3,
        im_size=[192, 192],
        in_ch=3,
        use_clip=True,
        device='cuda'
    ):
        super().__init__()
        self.k_dim = k_dim
        self.joint_dim = joint_dim
        self.in_ch = in_ch
        self.use_clip = use_clip

        activation = nn.LeakyReLU(negative_slope=0.3)

        sub_im_size = [
            im_size[0] - 3 * (kernel_size - 1),
            im_size[1] - 3 * (kernel_size - 1)
        ]
        self.sub_im_size = sub_im_size
        self.heatmap_size = heatmap_size

        if self.use_clip:
            self.pos_encoder = CLIPGuidedPosEncoder(k_dim=k_dim, temperature=temperature, device=device)
        else:
            self.pos_encoder = nn.Sequential(
                nn.Conv2d(self.in_ch, 16, 3, 1, 0), activation,
                nn.Conv2d(16, 32, 3, 1, 0), activation,
                nn.Conv2d(32, self.k_dim, 3, 1, 0), activation,
                SpatialSoftmax(width=sub_im_size[0], height=sub_im_size[1],
                               temperature=temperature, normalized=True)
            )

        self.im_encoder = nn.Sequential(
            nn.Conv2d(self.in_ch, 16, 3, 1, 0), activation,
            nn.Conv2d(16, 32, 3, 1, 0), activation,
            nn.Conv2d(32, self.k_dim, 3, 1, 0), activation,
        )

        rec_in = joint_dim + self.k_dim * 2
        self.rec = nn.LSTMCell(rec_in, rec_dim)

        self.decoder_joint = nn.Linear(rec_dim, joint_dim)
        self.decoder_point = nn.Linear(rec_dim, self.k_dim * 2)

        self.issm = InverseSpatialSoftmax(
            width=sub_im_size[1],
            height=sub_im_size[0],
            heatmap_size=self.heatmap_size,
            normalized=True
        )

        self.decoder_image = nn.Sequential(
            nn.ConvTranspose2d(self.k_dim, 32, 3, 1, 0), activation,
            nn.ConvTranspose2d(32, 16, 3, 1, 0), activation,
            nn.ConvTranspose2d(16, self.in_ch, 3, 1, 0), activation,
        )

        self.act = activation

    def encode_text(self, text_list):
        # Encode text via CLIP (optional).
        if not self.use_clip:
            return None

        text = clip.tokenize(text_list).to(next(self.parameters()).device)
        with torch.no_grad():
            text_features = self.pos_encoder.clip_model.encode_text(text)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.float()

    def forward(self, xi, xv, state=None, text_embedding=None):
        im_hid = self.im_encoder(xi)

        if self.use_clip:
            enc_pts, _ = self.pos_encoder(xi, text_embedding)
        else:
            enc_pts, _ = self.pos_encoder(xi)

        enc_pts = enc_pts.reshape(-1, self.k_dim * 2)

        hid = torch.cat([enc_pts, xv], -1)
        rnn_hid = self.rec(hid, state)

        y_joint = self.act(self.decoder_joint(rnn_hid[0]))
        dec_pts = self.act(self.decoder_point(rnn_hid[0]))

        dec_pts_in = dec_pts.reshape(-1, self.k_dim, 2)
        heatmap = self.issm(dec_pts_in)
        heatmap = heatmap.permute(0, 1, 3, 2)

        if heatmap.shape[2:] != im_hid.shape[2:]:
            heatmap = F.interpolate(heatmap,
                                    size=im_hid.shape[2:],
                                    mode="bilinear",
                                    align_corners=False)

        hid = heatmap * im_hid
        y_image = self.decoder_image(hid)

        return y_image, y_joint, enc_pts, dec_pts, rnn_hid

