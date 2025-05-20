import torch
from torch import nn
import torchvision.transforms as T
from utils_3D.transforms import GroupScale, GroupCenterCrop

class X3D(nn.Module):
    """
    X3D wrapper that can either extract features or train end-to-end:
      - If num_classes is None: forward() returns raw features (B, feat_dim)
      - If num_classes is set: forward() returns logits (B, num_classes)
    """
    def __init__(self,
                 size='l',
                 pretrained=True,
                 clip_len=16,
                 input_size=112,
                 num_classes=None):
        super().__init__()
        size = size.lower()
        assert size in ['xs','s','m','l'], f"Unsupported X3D size: {size}"
        hub_name = f'x3d_{size}'
        # 1) load the pretrained backbone and strip off its head
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo:main',
            hub_name,
            pretrained=pretrained
        )
        self.backbone.blocks[-1] = nn.Identity()

        # 2) metadata (for transforms)
        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.arch       = f"X3D-{size.upper()}"

        # 3) compute feature dim (channels out of backbone)
        dummy = torch.zeros(1, 3, clip_len, input_size, input_size)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_dim = feats.shape[1]

        # 4) if training, append a small classifier head; else leave classifier=None
        self.num_classes = num_classes
        if num_classes is not None:
            self.classifier = nn.Linear(self.feat_dim, num_classes)
        else:
            self.classifier = None

    def get_augmentation(self, crop_corners=True, do_horizontal_flip=True):
        return T.Compose([
            GroupScale(self.input_size),
            GroupCenterCrop(self.input_size)
        ])

    def forward(self, x):
        # ensure 5D: (B,C,T,H,W)
        if x.dim() == 4:
            x = x.unsqueeze(2)

        # 1) spatio-temporal features: (B, C, T', H', W')
        feats = self.backbone(x)

        # if no classifier head, just return the raw features
        if self.classifier is None:
            return feats

        # 2) global‐average‐pool → (B, C)
        pooled = feats.mean(dim=[2, 3, 4])

        # 3) compute logits → (B, num_classes)
        logits = self.classifier(pooled)

        # 4) return both logits and the pooled features
        return logits, pooled
