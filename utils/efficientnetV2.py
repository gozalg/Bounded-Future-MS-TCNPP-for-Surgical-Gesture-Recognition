import torch
from torch import nn
import timm
from torch.nn.modules.linear import Identity
import torchvision
from utils.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip

class EfficientnetV2(nn.Module):
    def __init__(self, size, num_classes, pretrained,input_size=224):
        super().__init__()
        self.input_size = input_size
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]
        self.arch = "EfficientNetV2"




        if size == "s":
            self.base = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=1,)
        elif size == "m":
            self.base = timm.create_model("tf_efficientnetv2_m", pretrained=pretrained, num_classes=1)
        elif size == "l":
            self.base  = timm.create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=1)
        else:
            raise NotImplementedError()

        self.base.classifier =Identity()

        penultimate_shape = 1280
        self.added_fc = nn.Linear(penultimate_shape, num_classes)

    def forward(self,x):
        features = self.base(x)
        output = self.added_fc(features)

        return output, features

    def get_augmentation(self, crop_corners=True, do_horizontal_flip=True):
        if do_horizontal_flip:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],
                                                                       fix_crop=crop_corners,
                                                                       more_fix_crop=crop_corners),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        else:
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66],
                                                                       fix_crop=crop_corners,
                                                                       more_fix_crop=crop_corners)])


