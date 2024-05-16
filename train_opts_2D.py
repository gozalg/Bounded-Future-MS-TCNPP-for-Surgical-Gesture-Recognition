import argparse
import os


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
current_dataset = 'JIGSAWS' # 'VTS' # 'MultiBypass140' # 'RARP50' #


parser = argparse.ArgumentParser(description="Train model for video-based surgical gesture recognition.")
parser.register('type', 'bool', str2bool)

# Experiment
parser.add_argument('--project_name', type=str, default=f"{current_dataset}_Feature_Extractor",
                    help="Name of the project in wandb.")
parser.add_argument('--test', type=str2bool, default=False,
                    help="Whether the run is a test")
parser.add_argument('--seed', type=int, default=42, help="Random seed.")
parser.add_argument('--exp', type=str, default=f"{current_dataset}_experiment",
                    help="Name (description) of the folder (parent folder) in which the experiment is run.")
parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOUO',
                    help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO)." + 
                    "Only LOUO supported for TBD.")
parser.add_argument('--image_tmpl', default='img_{:05d}.jpg')
parser.add_argument('--video_suffix', type=str,choices=['_capture1', # relevant for jigsaws
                                                        '_capture2', # relevant for jigsaws
                                                       ], default='_capture2')
# Data
parser.add_argument('--dataset', type=str, default=current_dataset, choices=['VTS', 'JIGSAWS', 'MultiBypass140', 'RARP50'],
                    help="Name of the dataset to use.")

parser.add_argument('--epoch_size', type=int, default=2400,
                    help="number of samples in a epoch ")
#------------------------ VTS ------------------------
if current_dataset=='VTS':
    parser.add_argument('--num_classes', type=int,
                        default=6, help="Number of classes.")
    parser.add_argument('--data_path', type=str, default=os.path.join(data_dir, current_dataset, "frames"),
                        help="Path to data folder, which contains the extracted images for each video. "
                             "One subfolder per video.")
    parser.add_argument('--transcriptions_dir', type=str,
                        default=os.path.join(data_dir, current_dataset, "transcriptions_gestures"),
                        help="Path to folder containing the transcription files (gesture annotations). One file per video.")
#---------------------- JIGSAWS ----------------------
elif current_dataset=='JIGSAWS':
    parser.add_argument('--num_classes', type=int,
                        default=10, help="Number of classes.")
    parser.add_argument('--data_path', type=str, default=os.path.join(data_dir, current_dataset, "Suturing", "frames"),
                        help="Path to data folder, which contains the extracted images for each video. "
                             "One subfolder per video.")
    parser.add_argument('--transcriptions_dir', type=str, default=os.path.join(data_dir, current_dataset, "Suturing", "transcriptions"),
                        help="Path to folder containing the transcription files (gesture annotations). One file per video.")
    parser.add_argument('--video_lists_dir', type=str, default=os.path.join(data_dir, current_dataset, "Splits", "Suturing"),
                    help="Path to directory containing information about each video in the form of video list files. "
                         "One subfolder per evaluation scheme, one file per evaluation fold.")
    parser.add_argument('--task', type=str, choices=['Suturing', 'Needle_Passing', 'Knot_Tying'], default='Suturing',
                    help="JIGSAWS task to evaluate.")
#------------------- MultiBypass140 -------------------
elif current_dataset=='MultiBypass140':
    raise NotImplementedError()
#----------------------- RARP50 -----------------------
elif current_dataset=='RARP50':
    raise NotImplementedError()

parser.add_argument('--video_sampling_step', type=int, default=1,
                    help="Describes how the available video data has been downsampled from the original temporal "
                         "resolution (by taking every <video_sampling_step>th frame).")
parser.add_argument('--do_horizontal_flip', type='bool', default=False,
                    help="Whether data augmentation should include a random horizontal flip.")
parser.add_argument('--do_vertical_flip', type='bool', default=False,
                    help="Whether data augmentation should include a random vertical flip.")
parser.add_argument('--do_color_jitter', type='bool', default=False,
                    help="Whether data augmentation should include a random jitter.")
parser.add_argument('--perspective_distortion', type=float, default=0, choices=Range(0.0, 1.0),
                    help="Argument to control the degree of distortion." 
                         "If 0 then augmentaion is not applied.")
parser.add_argument('--degrees', type=int, default=0,
                    help="Number of degrees for random roation augmenation."
                         "If 0 then augmentation is not applied")
parser.add_argument('--corner_cropping', type='bool', default=True,
                    help="Whether data augmentation should include corner cropping.")
parser.add_argument('--vae_intermediate_size', type=int, default=None,
                    help="VAE latent space dim")
parser.add_argument('--additional_param_num', type=int, default=0, 
                    help="Number of parameters in additional linear layer. if 0 then no additional layer is added to the model")
parser.add_argument('--decoder_weight', type=float, default=0, 
                    help="Weight of decoder loss.")
parser.add_argument('--certainty_weight', type=float, default=0, 
                    help="Weight of certainty loss.")
parser.add_argument('--word_embdding_weight', type=float, default=0, 
                    help="Weight of word embdding loss.")
parser.add_argument('--class_criterion_weight', type=float, default=1, 
                    help="Weight of class prediction loss")
parser.add_argument('--x_sigma2', type=float, default=1, 
                    help="Likelihood variance for vae")

parser.add_argument('--preload', type='bool', default=True,
                    help="Whether to preload all training set images before training")
parser.add_argument('--number_of_samples_per_class', type=int, default=400,
                    help="Number of samples taken from each class for training")

# Model

parser.add_argument('--arch', type=str, default="2D-EfficientNetV2-m", choices=['3D-ResNet-18', '3D-ResNet-50', "2D-ResNet-18", "2D-ResNet-34",
                                                                         "2D-EfficientNetV2-s", "2D-EfficientNetV2-m", 
                                                                         "2D-EfficientNetV2-l"],
                    help="Network architecture.")
parser.add_argument('--use_resnet_shortcut_type_B', type='bool', default=False,
                    help="Whether to use shortcut connections of type B.")
parser.add_argument('--snippet_length', type=int, default=1, help="Number of frames constituting one video snippet.")
parser.add_argument('--input_size', type=int, default=224,
                    help="Target size (width/ height) of each frame.")
# Training
parser.add_argument('--resume_exp', type=str, default=None,
                    help="Path to results of former experiment that shall be resumed (UNTESTED).")
parser.add_argument('-j', '--workers', type=int, default=16, help="Number of threads used for data loading.")
parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train.")
parser.add_argument('-b', '--batch_size', type=int, default=32, help="Batch size.")
parser.add_argument('--lr', type=float, default=0.00025, help="Learning rate.")
parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay.")
parser.add_argument('--gpu_id', type=int, default=0, help="Device id of gpu to use.")
parser.add_argument('--use_scheduler', type=bool, default=True, help="Whether to use the learning rate scheduler.")
parser.add_argument('--loss_weighting', type=bool, default=True,
                    help="Whether to apply weights to loss calculation so that errors in more current predictions "
                         "weigh more heavily.")
parser.add_argument('--eval_freq', '-ef', type=int, default=1, help="Validate model every <eval_freq> epochs.")
parser.add_argument('--save_freq', '-sf', type=int, default=5, help="Save checkpoint every <save_freq> epochs.")
parser.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output"),
                    help="Path to output folder, where all models and results will be stored.")
parser.add_argument('--label_embedding_path', type=str, default=None,
                    help="Path to label embeddings, where a vector embedding will be saved for each label")
parser.add_argument('--margin', type=float, default=1,
                    help="Word Embedding loss margin")
parser.add_argument('--positive_aggregator', type=str, default="max", choices=["max", "sum"],
                    help="Word Embedding loss positive_aggregator")
parser.add_argument('--split_num', type=int, default=None,
                    help="split number to use as validation set. If None, apply cross validation")
