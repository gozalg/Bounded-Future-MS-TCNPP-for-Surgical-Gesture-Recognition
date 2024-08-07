# parts of the code were adapted from: https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com
# imports
import torch
from Trainer import Trainer
from batch_gen2 import BatchGenerator
import os
import argparse
import pandas as pd
from datetime import datetime
from termcolor import colored, cprint
import random
from sklearn.model_selection import KFold

import time

from utils.util import AverageMeter, splits_LOSO, splits_LOUO, splits_LOUO_NP, gestures_SU, gestures_NP, gestures_KT
from utils.util import gestures_GTEA, splits_GTEA, splits_50salads, gestures_50salads, splits_breakfast, gestures_breakfast


def get_k_folds_splits(k=5, shuffle=True, args=None):

    if not args:
        args = parser.parse_args()

    if args.dataset == "JIGSAWS":
        users = splits_LOUO
    elif args.dataset == "GTEA":
        users = splits_GTEA
    elif args.dataset == "50SALADS":
        users = splits_50salads

    if shuffle:
        kfolds = KFold(n_splits=k, shuffle=shuffle, random_state=args.seed)
    else:
        kfolds = KFold(n_splits=k, shuffle=shuffle)

    for train, test in kfolds.split(users):
        split = [users[i] for i in test]
        yield split


def get_splits(dataset, eval_scheme, task):
    splits = None
    if dataset == "JIGSAWS":
        if eval_scheme == 'LOSO':
            splits = splits_LOSO
        elif eval_scheme == 'LOUO':
            if task == "Needle_Passing":
                splits = splits_LOUO_NP
            else:
                splits = splits_LOUO
    elif dataset == "GTEA":
        if eval_scheme == "LOUO":
            splits = splits_GTEA
    elif dataset == "50SALADS":
        if eval_scheme == "LOUO":
            splits = splits_50salads
            splits = list(get_k_folds_splits(k=5, shuffle=False))
    elif dataset == "BREAKFAST":
        if eval_scheme == "LOUO":
            splits = splits_breakfast

    return splits


def train_val_split(splits, val_split):

    if isinstance(val_split, int):
        assert (val_split >= 0 and val_split < len(splits))
        train_lists = splits[0:val_split] + splits[val_split + 1:]
        val_list = splits[val_split:val_split+1]

        if isinstance(train_lists[0], list):
            train_lists = [
                item for train_split in train_lists for item in train_split]
        if isinstance(val_list[0], list):
            val_list = [item for val_split in val_list for item in val_split]

    else:
        assert isinstance(val_split, list)
        assert all((s in splits) for s in val_split)
        train_lists = [s for s in splits if s not in val_split]
        val_list = [s for s in splits if s in val_split]

    return train_lists, val_list


data = "/data/"
dt_string = datetime.now().strftime("%d.%m.%Y %H:%M:%S")


parser = argparse.ArgumentParser()
parser.add_argument('--features_folder', type=str,
                    default="/data/home/ori.meiraz/Real-Online-MSTCN-/output-ori-new/features", help="the folder containing the features")
parser.add_argument('--transcriptions_dir', type=str, default=os.path.join("data", "Suturing", "transcriptions"),
                    help="Path to folder containing the transcription files (gesture annotations). One file per video.")
parser.add_argument('--Mode', type=int, default=1)
parser.add_argument('--arch', type=str, default="2D-ResNet-18", choices=['3D-ResNet-18', '3D-ResNet-50', "2D-ResNet-18", "2D-ResNet-34",
                                                                         "2D-EfficientNetV2-s", "2D-EfficientNetV2-m",
                                                                         "2D-EfficientNetV2-l"],
                    help="Network architecture.")

parser.add_argument('--dataset', choices=['VTS', 'GTEA'], default="VTS")
parser.add_argument(
    '--task', choices=['gestures', 'tools', 'multi-taks'], default="gestures")
parser.add_argument(
    '--network', choices=['MS-TCN2', 'MS-TCN2 late', 'MS-TCN2 early'], default="MS-TCN2")
parser.add_argument(
    '--split', choices=['0', '1', '2', '3', '4', 'all'], default='all')
parser.add_argument('--features_dim', default=1280, type=int)
parser.add_argument('--lr', default='0.0010351748096577', type=float)
parser.add_argument('--num_epochs', default=40, type=int)
parser.add_argument('--eval_rate', default=1, type=int)

# Architectuyre
parser.add_argument('--num_f_maps', default=128, type=int)

parser.add_argument('--normalization', choices=[
                    'Min-max', 'Standard', 'samplewise_SD', 'none'], default='none', type=str)

parser.add_argument('--sample_rate', default=1, type=int)
parser.add_argument('--offline_mode', default=False, type=bool)


parser.add_argument('--loss_tau', default=16, type=float)
parser.add_argument('--loss_lambda', default=1, type=float)
parser.add_argument('--dropout_TCN', default=0.5, type=float)
parser.add_argument(
    '--project', default="MS-TCN online- offline tradeoff", type=str)
parser.add_argument('--group', default=dt_string + " ", type=str)
parser.add_argument('--use_gpu_num', default="0", type=str)
parser.add_argument('--upload', default=False, type=bool)
parser.add_argument('--debagging', default=False, type=bool)
parser.add_argument('--hyper_parameter_tuning', default=False, type=bool)


args = parser.parse_args()
debagging = args.debagging
if debagging:
    args.upload = False

if args.Mode == 1:
    window_dim = 1

elif args.Mode == 2:
    window_dim = 3

elif args.Mode == 3:
    window_dim = 7

elif args.Mode == 4:
    window_dim = 10

elif args.Mode == 5:
    window_dim = 12

elif args.Mode == 6:
    window_dim = 15

elif args.Mode == 7:
    window_dim = 17

elif args.Mode == 8:
    window_dim = 0
    args.offline_mode = True

else:
    raise NotImplemented

Number_of_refinements = [0, 1, 2, 3]
Number_of_layers = [5, 4, 3, 2]

for n_ref in Number_of_refinements:
    # n_ref = number of refinements in current experiment/
    # meaning, there are n_ref refinement stages
    args.num_R = n_ref

    for n_l in Number_of_layers:
        # n_l = number of layers for each stage in current experiment
        args.split = "all"
        args.num_layers_PG = n_l  # number of layers in prediction generation
        args.num_layers_R = n_l  # number of layers in refinement

        print(args)
        # make deterministic
        seed = int(time.time())
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpu_num

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        # use the full temporal resolution @ 30Hz

        sample_rate = args.sample_rate
        bz = 2

        list_of_splits = []
        if len(args.split) == 1:
            list_of_splits.append(int(args.split))

        elif args.dataset == "VTS":
            list_of_splits = list(range(0, 5))
        elif args.dataset == 'GTEA':
            list_of_splits = list(range(0, len(splits_GTEA)))
        else:
            raise NotImplemented

        loss_lambda = args.loss_lambda
        loss_tau = args.loss_tau
        num_epochs = args.num_epochs
        eval_rate = args.eval_rate
        features_dim = args.features_dim
        lr = args.lr
        offline_mode = args.offline_mode
        num_layers_PG = args.num_layers_PG
        num_layers_R = args.num_layers_R
        num_f_maps = args.num_f_maps
        experiment_name = args.group + " task:" + args.task + " splits: " + args.split + " net: " + args.network +\
            " is Offline: " + str(args.offline_mode) + " window dim: " + \
            str(window_dim) + " num of layers: " + \
            str(n_l) + " num of ref stages: " + str(n_ref)

        hyper_parameter_tuning = args.hyper_parameter_tuning
        print(colored(experiment_name, "green"))

        summaries_dir = "./summaries/" + args.dataset + "/" + experiment_name
        if not debagging:
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)

        full_eval_results = pd.DataFrame()
        full_train_results = pd.DataFrame()
        full_test_results = pd.DataFrame()

        for split_num in list_of_splits:
            print("split number: " + str(split_num))
            args.split = str(split_num)

            folds_folder = os.path.join(data, args.dataset, "folds")

            if args.dataset == "VTS":
                features_path = os.path.join(
                    data, args.dataset, "features", "fold " + str(split_num))

            else:
                # features_path = "./data/" + args.dataset + "/kinematics_npy/"
                features_path = os.path.join(
                    args.features_folder, args.dataset, args.arch, "fold " + str(split_num))

            # gt_path_gestures = "./data/"+args.dataset+"/transcriptions_gestures/"
            gt_path_gestures = args.transcriptions_dir
            # gt_path_tools_left = "./data/"+args.dataset+"/transcriptions_tools_left/"
            gt_path_tools_left = os.path.join(
                data, args.dataset, "transcriptions_tools_left")
            # gt_path_tools_right = "./data/"+args.dataset+"/transcriptions_tools_right/"
            gt_path_tools_right = os.path.join(
                data, args.dataset, "transcriptions_tools_right")

            # mapping_gestures_file = "./data/"+args.dataset+"/mapping_gestures.txt"
            mapping_gestures_file = os.path.join(
                data, args.dataset, "mapping_gestures.txt")

            # mapping_tool_file = "./data/"+args.dataset+"/mapping_tools.txt"
            mapping_tool_file = os.path.join(
                data, args.dataset, "mapping_tools.txt")

            # model_dir = "./models/"+args.dataset+"/"+ experiment_name+"/split_"+args.split
            model_dir = os.path.join(
                "models", args.dataset, experiment_name, "split" + args.split)

            if not debagging:
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

            if args.dataset == 'VTS':
                file_ptr = open(mapping_gestures_file, 'r')
                # Example: ['0 T0', '1 T1'...]
                actions = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                actions_dict_gestures = dict()
                for a in actions:
                    actions_dict_gestures[a.split()[1]] = int(a.split()[0])
            elif args.dataset == 'GTEA':
                actions_dict_gestures = {
                    x: i for (i, x) in enumerate(gestures_GTEA)}

            num_classes_tools = 0
            actions_dict_tools = dict()
            if args.dataset == "VTS":
                file_ptr = open(mapping_tool_file, 'r')
                actions = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                for a in actions:
                    actions_dict_tools[a.split()[1]] = int(a.split()[0])
                # Example: actions_dict_tools = {'T0':0, 'T1":1...}
                num_classes_tools = len(actions_dict_tools)

            num_classes_gestures = len(actions_dict_gestures)

            if args.task == "gestures":
                num_classes_list = [num_classes_gestures]
            elif args.dataset == "VTS" and args.task == "tools":
                num_classes_list = [num_classes_tools, num_classes_tools]
            elif args.dataset == "VTS" and args.task == "multi-taks":
                num_classes_list = [num_classes_gestures,
                                    num_classes_tools, num_classes_tools]

            trainer = Trainer(num_layers_PG, num_layers_R, args.num_R, num_f_maps, features_dim, num_classes_list,
                              offline_mode=offline_mode, window_dim=window_dim,
                              tau=loss_tau, lambd=loss_lambda,
                              dropout_TCN=args.dropout_TCN, task=args.task, device=device,
                              network=args.network,
                              hyper_parameter_tuning=hyper_parameter_tuning, debagging=debagging, has_test=(args.dataset == "VTS"))
            print(gt_path_gestures)
            splits = get_splits(args.dataset, 'LOUO', args.task)
            train_lists, val_list = train_val_split(splits, int(args.split))

            lists_dir = os.path.join(
                '/data/home/orrubin/projects/2d_learnersv3/GTEA/Splits/', 'LOUO')
            train_lists = list(
                map(lambda x: os.path.join(lists_dir, x), train_lists))
            val_list = list(
                map(lambda x: os.path.join(lists_dir, x), val_list))

            batch_gen = BatchGenerator(num_classes_gestures, num_classes_tools, actions_dict_gestures, actions_dict_tools, features_path, split_num, folds_folder,
                                       gt_path_gestures, gt_path_tools_left, gt_path_tools_right, val_list=val_list, train_lists=train_lists, sample_rate=sample_rate, normalization=args.normalization, task=args.task)
            eval_dict = {"features_path": features_path, "actions_dict_gestures": actions_dict_gestures, "actions_dict_tools": actions_dict_tools, "device": device, "sample_rate": sample_rate, "eval_rate": eval_rate,
                         "gt_path_gestures": gt_path_gestures, "gt_path_tools_left": gt_path_tools_left, "gt_path_tools_right": gt_path_tools_right, "task": args.task}
            best_valid_results, eval_results, train_results, test_results = trainer.train(
                model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, eval_dict=eval_dict, args=args)

            if not debagging:
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
