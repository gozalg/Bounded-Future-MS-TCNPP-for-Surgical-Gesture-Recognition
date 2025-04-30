import torch
from torch import nn
import torchvision.transforms as T
from utils_3D.transforms import GroupScale, GroupCenterCrop

class X3D(nn.Module):
    """
    Wrapper for the pretrained X3D video backbone (XS, S, M, L, XL).
    Loads via Torch Hub, removes the classification head, and exposes feature extraction.
    """
    def __init__(self, size='l', pretrained=True, clip_len=16, input_size=112):
        super().__init__()
        # Validate size
        size = size.lower()
        if size not in ['xs', 's', 'm', 'l', 'xl']:
            raise ValueError(f"Unsupported X3D size: {size}")
        hub_name = f'x3d_{size}'
        # Load pretrained model from PyTorchVideo Hub
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo:main',
            hub_name,
            pretrained=pretrained
        )
        # Remove its final projection head so forward() yields raw features
        self.backbone.blocks[-1] = nn.Identity()

        # Store input and normalization metadata
        self.input_size = input_size
        self.input_mean = [0.45, 0.45, 0.45]
        self.input_std  = [0.225, 0.225, 0.225]
        self.arch       = f"X3D-{size.upper()}"

        # Determine feature dimension via dummy forward
        dummy = torch.zeros(1, 3, clip_len, input_size, input_size)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_dim = feats.shape[1]

    def get_augmentation(self, crop_corners=True, do_horizontal_flip=True):
        """Simple clip transforms: scale + center crop"""
        return T.Compose([
            GroupScale(self.input_size),
            GroupCenterCrop(self.input_size)
        ])

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W) or (B, C, T, H, W)
        returns: Tensor of shape (B, feat_dim)
        """
        # If given 4D input (frame), add a time dimension at position=2
        if x.dim() == 4:
            x = x.unsqueeze(2)  # becomes (B, C, 1, H, W)
        return self.backbone(x)
