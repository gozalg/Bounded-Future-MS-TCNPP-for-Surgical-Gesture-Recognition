import argparse
import os

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
current_dataset = 'JIGSAWS' # 'JIGSAWS' # 'SAR_RARP50' # 'VTS' # 'MultiBypass140' # 
current_server  = 'so-srv1' # 'DGX' , 'so-srv1' , 'so1'


parser = argparse.ArgumentParser(description="Train model for video-based surgical gesture recognition.")
parser.register('type', 'bool', str2bool)
# ----------------------
# Experiment
# ----------------------
# Added - from here
parser.add_argument('--wandb', type=str2bool, default=False,
                    help="Whether to use wandb for logging.")
parser.add_argument('--project_name', type=str, default=f"{current_dataset}_Feature_Extractor_{current_server}",
                    help="Name of the project in wandb.")
parser.add_argument('--test', type=str2bool, default=False,
                    help="Whether the run is a test")
parser.add_argument('--seed', type=int, default=42, help="Random seed.") # 1538574472
# Added - till here
parser.add_argument('--exp', type=str, default=f"{current_dataset}",
                    help="Name (description) of the folder (parent folder) in which the experiment is run.")
# ----------------------
# Data
# ----------------------
# Added - from here
parser.add_argument('--dataset', type=str, default=current_dataset, choices=['VTS', 'JIGSAWS', 'MultiBypass140', 'SAR_RARP50'],
                    help="Name of the dataset to use.")
parser.add_argument('--data_path', type=str, default=os.path.join(data_dir, current_dataset, "frames"),
                        help="Path to data folder, which contains the extracted images for each video. "
                             "One subfolder per video.")
parser.add_argument('--transcriptions_dir', type=str, default=os.path.join(data_dir, current_dataset, "transcriptions"),
                    help="Path to folder containing the transcription files (gesture\steps\phases annotations). One file per video.")
parser.add_argument('--video_lists_dir', type=str, default=os.path.join(data_dir, current_dataset, "Splits"),
               help="Path to directory containing information about each video in the form of video list files. "
                    "One subfolder per evaluation scheme, one file per evaluation fold.")
# parser.add_argument('--epoch_size', type=int, default=2400,
#                     help="number of samples in a epoch ")
parser.add_argument('--number_of_samples_per_class', type=int, default=400,
                    help="Number of samples taken from each class for training")
# Added - till here
#-------------------------------------------------------------
#------------------------ VTS ------------------------
if current_dataset=='VTS':
    # Added - from here
    parser.add_argument('--num_classes', type=int, default=6, 
                        help="Number of classes.")
    # Added - till here
    parser.add_argument('--image_tmpl', default='img_{:05d}.jpg')
    parser.add_argument('--video_suffix', type=str,choices=['_side', '_top', # relevant for VTS
                                                            '_capture1', '_capture2' # relevant for JIGSAWS
                                                            'None'], default='_side')
    parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying', 'gesture', 'steps', 'phases'], default='gesture',
                        help =  "['Suturing', 'Needle_Passing', 'Knot_Tying'] - JIGSAWS task to evaluate.\n" +
                                "['steps', 'phases'] - MultiBypass140 task to evaluate.\n" +
                                "'gesture' - VTS & SAR_RARP50 task to evaluate.")
    parser.add_argument('--val_sampling_step', type=int, default=1,
                    help="Describes how the validation video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
#---------------------- JIGSAWS ----------------------
elif current_dataset=='JIGSAWS':
#     raise NotImplementedError
    # Added - from here
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="Number of classes.")
    # Added - till here
    parser.add_argument('--image_tmpl', default='img_{:05d}.jpg')
    parser.add_argument('--video_suffix', type=str,choices=['_side', '_top', # relevant for VTS     
                                                            '_capture1', '_capture2' # relevant for JIGSAWS
                                                            'None'], default='_capture2')
    parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying', 'gesture', 'steps', 'phases'], default='Suturing',
                        help =  "['Suturing', 'Needle_Passing', 'Knot_Tying'] - JIGSAWS task to evaluate.\n" +
                                "['steps', 'phases'] - MultiBypass140 task to evaluate.\n" +
                                "'gesture' - VTS & SAR_RARP50 task to evaluate.")
    parser.add_argument('--val_sampling_step', type=int, default=80,
                    help="Describes how the validation video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
#------------------- MultiBypass140 -------------------
elif current_dataset=='MultiBypass140':
    raise NotImplementedError
    # Added - from here
    parser.add_argument('--num_classes', type=int, default=14, 
                        help="Number of classes.") # 14 for phases, 46 for steps
    # Added - till here
    parser.add_argument('--image_tmpl', default='{}_{:08d}.jpg') # 1st arg is dir name, 2nd arg is frame number
    parser.add_argument('--video_suffix', type=str,choices=['_side', '_top', # relevant for VTS     
                                                            '_capture1', '_capture2' # relevant for JIGSAWS
                                                            'None'], default='None')
    parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying', 'gesture', 'steps', 'phases'], default='phases',
                        help =  "['Suturing', 'Needle_Passing', 'Knot_Tying'] - JIGSAWS task to evaluate.\n" +
                                "['steps', 'phases'] - MultiBypass140 task to evaluate.\n" +
                                "'gesture' - VTS & SAR_RARP50 task to evaluate.")
    parser.add_argument('--val_sampling_step', type=int, default=30,
                    help="Describes how the validation video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
#--------------------- SAR_RARP50 ---------------------
elif current_dataset=='SAR_RARP50':
    raise NotImplementedError
    # Added - from here
    parser.add_argument('--num_classes', type=int, default=8, 
                        help="Number of classes.")
    # Added - till here
    parser.add_argument('--image_tmpl', default='{:09d}.png')
    parser.add_argument('--video_suffix', type=str,choices=['_side', '_top', # relevant for VTS     
                                                            '_capture1', '_capture2' # relevant for JIGSAWS
                                                            'None'], default='None')
    parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying', 'gesture', 'steps', 'phases'], default='gesture',
                        help =  "['Suturing', 'Needle_Passing', 'Knot_Tying'] - JIGSAWS task to evaluate.\n" +
                                "['steps', 'phases'] - MultiBypass140 task to evaluate.\n" +
                                "'gesture' - VTS & SAR_RARP50 task to evaluate.")
    parser.add_argument('--val_sampling_step', type=int, default=60,
                    help="Describes how the validation video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
#-------------------------------------------------------------
parser.add_argument('--video_sampling_step', type=int, default=1, # 6
                    help="Describes how the available video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
# ----------------------
# Augmentations
# ----------------------

parser.add_argument('--do_horizontal_flip', type='bool', default=True,
                    help="Whether data augmentation should include a random horizontal flip.")
parser.add_argument('--corner_cropping', type='bool', default=True,
                    help="Whether data augmentation should include corner cropping.")
# ----------------------
# Model
# ----------------------
parser.add_argument('--arch', type=str, default="EfficientnetV2", choices=['3D-ResNet-18', '3D-ResNet-50',"2D-ResNet-18","EfficientnetV2"],
                    help="Network architecture.")
parser.add_argument('--use_resnet_shortcut_type_B', type='bool', default=False,
                    help="Whether to use shortcut connections of type B.")
parser.add_argument('--input_size', type=int, default=224, help="Target size (width/ height) of each frame.")
# ----------------------
# Training
# ----------------------
parser.add_argument('--split_num', type=int, default=None,
                    help="split number to use as validation set. If None, apply cross validation")
parser.add_argument('--eval_freq', '-ef', type=int, default=3, help="Validate model every <eval_freq> epochs.")
parser.add_argument('--epochs', type=int, default=100, help="Number of epochs to train.")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size.")
parser.add_argument('-j', '--workers', type=int, default=48, help="Number of threads used for data loading.")
#  Adam optimizer
parser.add_argument('--lr', type=float, default=0.00025, help="Learning rate.")
parser.add_argument('--use_scheduler', type=bool, default=True, help="Whether to use the learning rate scheduler.")
#  TBD
# parser.add_argument('--loss_weighting', type=bool, default=True,
#                     help="Whether to apply weights to loss calculation so that errors in more current predictions "
#                          "weigh more heavily.")
# ----------------------
# Output
# ----------------------
parser.add_argument('--resume_exp', type=str, default=None,
                    help="Path to results of former experiment that shall be resumed (UNTESTED).")
parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "feature_extractor"),
                    help="Path to output folder, where all models and results will be stored.")