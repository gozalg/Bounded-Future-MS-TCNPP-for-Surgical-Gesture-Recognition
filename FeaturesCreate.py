
# Standard library imports
from logging import raiseExceptions
import os
import shutil
import sys
import argparse

# Third-party library imports
from PIL import Image
import numpy as np
import torch
from torch._C import device
import torch.utils.data as data
import torchvision
import torch.nn as nn

# Local application/library specific imports
from utils.transforms import GroupScale, GroupCenterCrop, GroupNormalize
from utils.train_dataset import rotate_snippet, Add_Gaussian_Noise_to_snippet
from utils.util import splits_LOSO, splits_LOUO, splits_LOUO_NP, splits_GTEA, splits_50salads
from FeatureExtractorTrainer import INPUT_MEAN, INPUT_STD, get_gestures, get_k_folds_splits, load_model

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
current_dataset = 'JIGSAWS' # 'VTS' # 'MultiBypass140' # 'RARP50' #
feature_extrractor = '2D-EfficientNetV2-m'

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_args():

    parser = argparse.ArgumentParser(description="create 2d features for video-based gesture recognition.")
    parser.register('type', 'bool', str2bool)
    # parser.add_argument('--name', type=str, default=f'{current_dataset}_base',
    #                     help="name of features")
    # Experiment
    parser.add_argument('--dataset', type=str, default=current_dataset, choices=['VTS', 'JIGSAWS', 'MultiBypass140', 'RARP50'],
                        help="Name of the dataset to use.")
    parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying'], default='Suturing',
                        help="JIGSAWS task to evaluate.")
    parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOSO',
                        help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO)." + 
                             "Only LOUO supported for GTEA and 50SALADS.")
    parser.add_argument('--video_suffix', type=str,choices=['_capture1', '_capture2'], default='_capture2')
    parser.add_argument('--image_tmpl', default='img_{:05d}.jpg')

    # Data
    parser.add_argument('--data_path', type=str, default=os.path.join(data_dir, current_dataset, "Suturing", "frames"),
                        help="Path to data folder, which contains the extracted images for each video. "
                        "One subfolder per video.")
    parser.add_argument('--video_lists_dir', type=str, default=os.path.join(data_dir, current_dataset, "Splits", "Suturing"),
                    help="Path to directory containing information about each video in the form of video list files. "
                         "One subfolder per evaluation scheme, one file per evaluation fold.")
    # Model
    parser.add_argument('--arch', type=str, default=feature_extrractor, choices=['3D-ResNet-18', '3D-ResNet-50', "2D-ResNet-18", "2D-ResNet-34",
                                                                            "2D-EfficientNetV2-s", "2D-EfficientNetV2-m", 
                                                                            "2D-EfficientNetV2-l"],
                        help="Network architecture.")

    parser.add_argument('--additional_param_num', type=int, default=0, 
                    help="number of parameters in additional linear layer. if 0 then no additional layer is added to the model")

    
    # parser.add_argument('-j', '--workers', type=int, default=48, help="Number of threads used for data loading.")
    parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size.")
    parser.add_argument('--input_size', type=int, default=224,
                    help="Target size (width/ height) of each frame.")
    parser.add_argument('--pretrain_path', type=str, required=False, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'models', current_dataset, feature_extrractor),
                        help="Path to root folder containing pretrained models weights")
    parser.add_argument('--gpu_id', type=int, default=0, help="Device id of gpu to use.")
    parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output',  'features', current_dataset, feature_extrractor),
                        help="Path to output folder, where all logs and features will be stored.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")

    return parser.parse_args()

class Sequential2DImageReader(data.Dataset):

    def __init__(self, root_path, video_id, frame_count,
                 snippet_length=16, sampling_step = 6,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_capture2",
                 return_3D_tensor=True, return_dense_labels=True,
                 transform=None, normalize=None, resize=224, initial_frame_idx=1):

        self.root_path = root_path
        self.video_name = video_id
        self.video_id = video_id
        # self.transcriptions_dir = transcriptions_dir
        # self.gesture_ids = gesture_ids
        self.snippet_length = snippet_length
        self.sampling_step = sampling_step
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_dense_labels = return_dense_labels #
        self.transform = transform
        self.normalize = normalize
        self.resize = resize
        self.frame_count = int(frame_count)
        self.initial_frame_idx = initial_frame_idx

        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.frame_num_data = {}
        # self.labels_data = {}
        self.frame_num_data[video_id] = list(range(initial_frame_idx, initial_frame_idx + frame_count, self.sampling_step))
        self._preload_images(video_id)


    def _preload_images(self, video_id):
        print("Preloading images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        
        for idx in self.frame_num_data[self.video_name]:
            imgs = self._load_image(img_dir, idx)
            images.extend(imgs)
        self.image_data[video_id] = images

    def _load_image(self, directory, idx):
        img = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
        img = torchvision.transforms.Resize((self.resize, self.resize))(img)
        return [img]

    def __getitem__(self, index):
        # frame_list = [index]
        # target =self._get_snippet_labels(self.video_name,frame_list)
        # target = target[-1]
        data = self.get_snippet(self.video_name, index)

        return data, index

    # def _get_snippet_labels(self, video_id, frame_list):
    #     assert self.return_dense_labels
    #     labels = self.labels_data[video_id]
    #     target = []
    #     idx = frame_list[-1]
    #     target.append(int(labels[idx]))
    #     return torch.tensor(target, dtype=torch.long)

    def get_snippet(self, video_id, idx):
        snippet = list()
        _idx = max(idx, 0)  # padding if required
        img = self.image_data[video_id][_idx]
        snippet.append(img)
        snippet = rotate_snippet(snippet,0.5)
        Add_Gaussian_Noise_to_snippet(snippet)
        snippet = [torchvision.transforms.Resize((self.resize, self.resize))(img) for img in snippet]
        snippet = self.transform(snippet)
        snippet = [torchvision.transforms.ToTensor()(img) for img in snippet]
        snippet = snippet[0]
        snippet = self.normalize(snippet)
        data = snippet
        return data


    def __len__(self):
        return (len(self.image_data[self.video_name])) -(self.snippet_length -1)



def extract_features_to_files(model, frames_loader, output_folder, save_template="{}.pt"):

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        
        for batch_i, (data, idx) in enumerate(frames_loader):
            data = data.to(device)
            output = model(data)

            bs = data.shape[0]

            for i in range(bs):
                save_path = os.path.join(output_folder, save_template.format(idx[i])) 
                torch.save(output[i].clone(), save_path)

def run_model(model, frames_loader):

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():

        out = []

        for batch_i, (data, idx) in enumerate(frames_loader):
            data = data.to(device)
            output = model(data)
            out.append(output)
        
        out = torch.cat(out, dim=0)



    return out.detach().T


        



def get_dataloaders(val_lists, data_path, image_tmpl, video_suffix, batch_size, input_size, workers):
    normalize = GroupNormalize(INPUT_MEAN, INPUT_STD)
    val_augmentation = torchvision.transforms.Compose([GroupScale(input_size),
                                                       GroupCenterCrop(input_size)])   ## need to be corrected
    
    val_videos = list()
    for list_file in val_lists:
        val_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
    val_loaders = list()
    for video in val_videos:

        dataset = Sequential2DImageReader(root_path=data_path, 
                                          video_id=video[0],
                                          frame_count=int(video[1]),
                                          snippet_length=1,
                                          sampling_step=1,
                                          image_tmpl=image_tmpl,
                                          video_suffix=video_suffix,
                                          normalize=normalize,
                                          resize=input_size,
                                          transform=val_augmentation)


        yield dataset.video_name, torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                              shuffle=False, num_workers=workers)




def feature_creator_split(weights_path, output_dir):

    args = get_args()

    model = load_model(arch=args.arch, weights_path=weights_path,
                       add_layer_param_num=args.additional_param_num)
    model.to(f"cuda:{args.gpu_id}")

    splits = None
    if args.dataset == "JIGSAWS":
        if args.eval_scheme == 'LOSO':
            splits = splits_LOSO
        elif args.eval_scheme == 'LOUO':
            if args.task == "Needle_Passing":
                splits = splits_LOUO_NP
            else:
                splits = splits_LOUO
    else:
        raise NotImplementedError()
    # elif args.dataset == "GTEA":
    #     if args.eval_scheme == "LOUO":
    #         splits = splits_GTEA
    # elif args.dataset == "50SALADS":
    #     if args.eval_scheme == "LOUO":
    #         splits = splits_50salads

    val_list = splits
    # if isinstance(split, int):
    #     assert (split >= 0 and split < len(splits))
    #     val_list = splits[split:split+1]
    # elif isinstance(split, List):
    #     assert all((s in splits) for s in split)
    #     val_list = [s for s in splits if s in split]
    # else:
    #     raise NotImplementedError("split parameter must by of type int or list")

    lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)

    val_lists = list(map(lambda x: os.path.join(lists_dir, x), val_list))


    val_loaders = get_dataloaders(val_lists, args.data_path, args.image_tmpl,
                                  args.video_suffix, args.batch_size, args.input_size, workers=1)

    for video_name, val_loader in val_loaders:

        path=os.path.join(output_dir, video_name + ".npy")
        features = run_model(model, val_loader).cpu().numpy()
        
        path = os.path.join(output_dir, video_name)
        np.save(path, features)

def run_feature_creator(get_split):
    """function creates features based on the args recived

    :param get_split: a function that parses file names and return the split if the file
    is a weight file relevent for the split and None to skip the file
    """    
    args = get_args()

    # if args.dataset == "JIGSAWS":
    #     user_num = len(splits_LOUO)
    # elif args.dataset == "GTEA":
    #     user_num = len(splits_GTEA)
    # elif args.dataset == "50SALADS":
    #     user_num = len(splits_50salads)

    out_dir = os.path.join(args.out, args.dataset, args.eval_scheme)
    
    if os.path.isdir(out_dir):
        val = None
        while val not in ['y', 'n']:
            print(out_dir)
            val = input("features exists, press y to override or n to revert:\n")
            if val == 'y':
                shutil.rmtree(out_dir)
            elif val == 'n':
                sys.exit()

    os.makedirs(out_dir)


    for root, dirs, _ in os.walk(os.path.join(args.pretrain_path, args.eval_scheme)):
        dirs.sort()
        for dir in dirs:
            for filename in os.walk(os.path.join(root, dir)).__next__()[2]:
                split = get_split(filename)

                if split is None:
                    continue
                # if split.isnumeric():
                #     split = int(split)
                else:
                    # split = split[1: -1].split(", ")
                    split = [int(x) if isinstance(x, str) and is_integer(x) else x for x in split if isinstance(x, int) or (isinstance(x, str) and is_integer(x))]
                    split = split[0] if len(split) == 1 else split

                print(f"split {split} started")

                full_path = os.path.join(root, dir, filename)

                split_dir = os.path.join(out_dir, str(split))

                os.makedirs(split_dir)

                feature_creator_split(weights_path=full_path, output_dir=split_dir) 
    

if __name__ == "__main__":

    # get_split = lambda filename:  "".join(os.path.splitext(filename)[0].split("_")[1:]).replace("'", "") if "best_" in filename else None 
    # get_split = lambda filename:  os.path.splitext(filename)[0][5:].replace("'", "") if "best_[" in filename else None 
    
    # get_split = lambda filename:  (os.path.splitext(filename)[0].split("_")[1:]) if "best_" in filename else None 
    get_split = lambda filename:  (os.path.splitext(filename)[0].split("_")[1:]) if "best_" in filename and os.path.splitext(filename)[0].split("_")[1].isdigit() else None


    run_feature_creator(get_split)



