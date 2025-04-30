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
    def __init__(self, size='l', pretrained=True, clip_len=16, input_size=112, num_classes=None):
        super().__init__()
        size = size.lower()
        assert size in ['xs','s','m','l','xl'], f"Unsupported X3D size: {size}"
        hub_name = f'x3d_{size}'
        # load the pretrained backbone
        self.backbone = torch.hub.load(
            'facebookresearch/pytorchvideo:main',
            hub_name,
            pretrained=pretrained
        )
        # strip off its original head
        self.backbone.blocks[-1] = nn.Identity()

        # metadata (for transforms)
        self.input_size = input_size
        self.input_mean = [0.45, 0.45, 0.45]
        self.input_std  = [0.225,0.225,0.225]
        self.arch       = f"X3D-{size.upper()}"

        # compute feature dim
        dummy = torch.zeros(1, 3, clip_len, input_size, input_size)
        with torch.no_grad():
            feats = self.backbone(dummy)
        self.feat_dim = feats.shape[1]

        # if training, append a small classifier
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
        # accept 4D or 5D input
        if x.dim()==4:
            x = x.unsqueeze(2)  # B,C -> B,C,1
        feats = self.backbone(x)      # (B, feat_dim)
        if self.classifier:
            return self.classifier(feats)  # (B, num_classes)
        return feats
