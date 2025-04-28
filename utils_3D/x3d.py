import torch
from pytorchvideo.models.efficient_x3d import create_efficient_x3d

def load_x3d_l(device):
    model = create_efficient_x3d(
        model_num_classes=1,
        expand_ratio='L',
        head_activation='identity'
    ).eval().to(device)
    ckpt = torch.hub.load_state_dict_from_url(
        "https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/efficient_x3d_l_original_form.pyth",
        map_location=device
    )
    model.load_state_dict(ckpt["model"])
    return model, model.head.fc.in_features
