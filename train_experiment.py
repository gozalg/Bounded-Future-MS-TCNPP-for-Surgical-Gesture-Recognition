# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
#----------------- Python Libraries Imports -----------------#
# Standard library imports
import argparse
from datetime import datetime
from logging import raiseExceptions
import os
import random
import time

# Third party imports
import pandas as pd
import torch
from termcolor import colored, cprint
#------------------ Bounded Future Imports ------------------#
# Local application imports
from batch_gen import BatchGenerator
from utils.Trainer import Trainer
#------------------------------------------------------------#

date_str = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

# set the args for the experiment
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MultiBypass140', choices=['VTS', 'JIGSAWS', 'MultiBypass140', 'SAR_RARP50'],
                    help="Name of the dataset to use.")
parser.add_argument('--eval_scheme', type=str, choices=['LOSO', 'LOUO'], default='LOUO',
                    help="Cross-validation scheme to use: Leave one supertrial out (LOSO) or Leave one user out (LOUO)." + 
                    "Only LOUO supported for TBD.")
parser.add_argument('--task', default="phases", choices=['gesture', 'steps', 'phases', 'tools', 'multi-taks'])
parser.add_argument('--feature_extractor', type=str, default="2D-EfficientNetV2-m", 
                    choices=['3D-ResNet-18', '3D-ResNet-50', 
                             "2D-ResNet-18", "2D-ResNet-34",
                             "2D-EfficientNetV2-s", "2D-EfficientNetV2-m", "2D-EfficientNetV2-l"])
parser.add_argument('--network', choices=['MS-TCN2', 'MS-TCN2 late', 'MS-TCN2 early'], default="MS-TCN2")
parser.add_argument('--split', choices=['0', '1', '2', '3', '4', '5', '6', '7', 'all'], default='all')
parser.add_argument('--features_dim', default=1280, type=int)
parser.add_argument('--lr', default='0.0010351748096577', type=float)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--eval_rate', default=1, type=int)

# Architecture
parser.add_argument('--w_max', default=17, type=int) # Relevant for BF-MS-TCN: 0 for "fully online".
parser.add_argument('--num_layers_PG', default=10, type=int) 
parser.add_argument('--num_layers_R', default=10, type=int)
parser.add_argument('--num_f_maps', default=128, type=int)

parser.add_argument('--normalization', choices=['Min-max', 'Standard', 'samplewise_SD', 'None'], default='None', type=str)
parser.add_argument('--num_R', default=3, type=int)

parser.add_argument('--sample_rate', default=1, type=int)
parser.add_argument('--RR_or_BF_mode', default="RR", type=str, choices=["RR", "BF"]) #True for RR-MS-TCN ("offline"), False for BF-MS-TCN ("online")


parser.add_argument('--loss_tau', default=16, type=float)
parser.add_argument('--loss_lambda', default=1, type=float)
parser.add_argument('--dropout_TCN', default=0.5, type=float)
parser.add_argument('--project', default="RR-MS-TCN_JIGSAWS_LOUO_wmax=0_so01", type=str) # default="Offline RNN nets Sensor paper Final"
parser.add_argument('--group', default=date_str + " ", type=str)
parser.add_argument('--use_gpu_num', default="1", type=str)
parser.add_argument('--upload', default=False, type=bool)
parser.add_argument('--DEBUG', default=False, type=bool)
parser.add_argument('--hyper_parameter_tuning', default=False, type=bool)

args = parser.parse_args()

DEBUG = args.DEBUG
if DEBUG:
    args.upload = False


print(args)
seed = int(time.time())  # setting seed for deterministic output
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num  # number of GPUs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

if device.type=='cpu':
    raiseExceptions("Not using CUDA")

# use the full temporal resolution @ 30Hz
sample_rate = args.sample_rate
batch_size = 2

list_of_splits = []
if args.split.isdigit(): # if split isn't 'all'
    list_of_splits.append(int(args.split))
elif args.dataset in ["VTS", "SAR_RARP50", "MultiBypass140"]:
    list_of_splits = list(range(0, 5))
elif args.dataset == "JIGSAWS":
    if args.eval_scheme == "LOUO":
        list_of_splits = list(range(0, 8))
    if args.eval_scheme == "LOSO":
        list_of_splits = list(range(0, 5))
else:
    raise NotImplementedError()

# the args for the model
loss_lambda = args.loss_lambda
loss_tau = args.loss_tau
num_epochs = args.num_epochs
eval_rate = args.eval_rate
features_dim = args.features_dim
lr = args.lr
RR_not_BF_mode = True if args.RR_or_BF_mode == "RR" else False
num_layers_PG = args.num_layers_PG
num_layers_R = args.num_layers_R
num_f_maps = args.num_f_maps
experiment_name = args.group + " task:" + args.task + " splits: " + args.split + " net: " + \
                  args.network + " is RR_or_BF_mode: " + str(args.RR_or_BF_mode) + " w_max: " + str(args.w_max)
args.group = experiment_name
hyper_parameter_tuning = args.hyper_parameter_tuning
print(colored(experiment_name, "green"))


summaries_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "summaries", args.dataset, args.eval_scheme, args.task, experiment_name) 


if not DEBUG:
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)

# create empty dataframes
full_eval_results = pd.DataFrame()
full_train_results = pd.DataFrame()
full_test_results = pd.DataFrame()

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
models = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "models", args.dataset, args.network, args.eval_scheme, args.task)
for split_num in list_of_splits:
    features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "features", args.dataset, args.feature_extractor, args.eval_scheme, args.task)
    print("split number: " + str(split_num))
    args.split = str(split_num)

    gt_path_gestures = os.path.join(data_dir, args.dataset, "transcriptions")
    if args.dataset == "MultiBypass140":
        mapping_gestures_file = os.path.join(data_dir, args.dataset, f"mapping_{args.task}.txt")
    else:
        mapping_gestures_file = os.path.join(data_dir, args.dataset, "mapping_gestures.txt")
    model_out_dir = os.path.join(models, experiment_name, "split" + args.split)

    if args.dataset == "VTS":
        raise NotImplementedError()
        features_path = os.path.join(data_dir, args.dataset, "features", "fold " + str(split_num))
        folds_dir = os.path.join(data_dir, args.dataset, "folds")
    elif args.dataset == "JIGSAWS":
        features_path = os.path.join(features_path, args.split)
        folds_dir = os.path.join(data_dir, args.dataset, "folds", args.eval_scheme)
    elif args.dataset in ["SAR_RARP50", "MultiBypass140"]:
        features_path = os.path.join(features_path, args.split)
        folds_dir = os.path.join(data_dir, args.dataset, "folds")
    else:
        raise NotImplementedError()

    if not DEBUG:
        if not os.path.exists(model_out_dir):
            os.makedirs(model_out_dir)

    file_ptr = open(mapping_gestures_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict_gestures = dict()
    for a in actions:
        actions_dict_gestures[a.split()[1]] = int(a.split()[0])
    num_classes_tools = 0
    # Example: {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3}
    actions_dict_tools = dict()
    # if args.dataset == "VTS":
    #     file_ptr = open(mapping_tool_file, 'r')
    #     actions = file_ptr.read().split('\n')[:-1]
    #     file_ptr.close()
    #     for a in actions:
    #         actions_dict_tools[a.split()[1]] = int(a.split()[0])
    #     num_classes_tools = len(actions_dict_tools)

    num_classes_gestures = len(actions_dict_gestures)

    if args.task in ["gesture", "steps", "phases"]:
        num_classes_list = [num_classes_gestures]
    elif args.dataset == "VTS" and args.task == "tools":
        num_classes_list = [num_classes_tools, num_classes_tools]
    elif args.dataset == "VTS" and args.task == "multi-taks":
        num_classes_list = [num_classes_gestures,
                            num_classes_tools, num_classes_tools]

    # initializes the Trainer - does not train
    trainer = Trainer(num_layers_PG, 
                      num_layers_R, 
                      args.num_R, 
                      num_f_maps, 
                      features_dim, 
                      num_classes_list,
                      RR_not_BF_mode            = RR_not_BF_mode, 
                      w_max                     = args.w_max,
                      tau                       = loss_tau, 
                      lambd                     = loss_lambda,
                      dropout_TCN               = args.dropout_TCN, 
                      task                      = args.task, 
                      device                    = device,
                      network                   = args.network,
                      hyper_parameter_tuning    = hyper_parameter_tuning, 
                      DEBUG                     = DEBUG)
    print(num_classes_gestures, 
          num_classes_tools, 
          actions_dict_gestures, 
          actions_dict_tools, 
          features_path, 
          split_num, 
          folds_dir,
          gt_path_gestures, 
          sample_rate, 
          args.normalization, 
          args.task)
    batch_gen = BatchGenerator(args.dataset, 
                               num_classes_gestures, 
                               num_classes_tools, 
                               actions_dict_gestures, 
                               actions_dict_tools, 
                               features_path, 
                               split_num, 
                               folds_dir, 
                               gt_path_gestures, 
                               sample_rate      = sample_rate, 
                               normalization    = args.normalization, 
                               task             = args.task)
    eval_dict = {"features_path": features_path, 
                 "actions_dict_gestures": actions_dict_gestures, 
                 "actions_dict_tools": actions_dict_tools, 
                 "device": device, 
                 "sample_rate": sample_rate, 
                 "eval_rate": eval_rate,
                 "gt_path_gestures": gt_path_gestures, 
                 "task": args.task}
    best_valid_results, eval_results, train_results, test_results = trainer.train(model_out_dir, 
                                                                                  summaries_dir,
                                                                                  batch_gen, 
                                                                                  num_epochs    = num_epochs, 
                                                                                  batch_size    = batch_size, 
                                                                                  learning_rate = lr, 
                                                                                  eval_dict     = eval_dict, 
                                                                                  args          = args)

    if not DEBUG:
        eval_results = pd.DataFrame(eval_results)
        train_results = pd.DataFrame(train_results)
        test_results = pd.DataFrame(test_results)
        eval_results["split_num"] = str(split_num)
        train_results["split_num"] = str(split_num)
        test_results["split_num"] = str(split_num)
        eval_results["seed"] = str(seed)
        train_results["seed"] = str(seed)
        test_results["seed"] = str(seed)

        full_eval_results = pd.concat(
            [full_eval_results, eval_results], axis=0)
        full_train_results = pd.concat(
            [full_train_results, train_results], axis=0)
        full_test_results = pd.concat(
            [full_test_results, test_results], axis=0)
        full_eval_results.to_csv(
            summaries_dir+"/"+args.network + "_evaluation_results.csv", index=False)
        full_train_results.to_csv(
            summaries_dir+"/"+args.network + "_train_results.csv", index=False)
        full_test_results.to_csv(
            summaries_dir+"/"+args.network + "_test_results.csv", index=False)
