# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology
#----------------- Python Libraries Imports -----------------#
# Python Standard Library
from copyreg import pickle
import datetime
import os
import os.path
import random
import string
import sys
sys.path.insert(0, os.path.abspath('..'))
from functools import partial
from operator import is_
from types import MethodType
from typing import Dict, List, Sequence

# Third-party libraries
import numpy as np
import pandas as pd
import PIL
import timm
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.nn.modules.linear import Identity
import torchvision
from torchvision.transforms.functional import InterpolationMode
from timm.models.efficientnet import EfficientNet
import shutil
from sklearn.model_selection import KFold
import tqdm
import wandb
# import optuna
#------------------ Bounded Future Imports ------------------#
from train_opts_2D import parser
from utils.resnet2D import resnet18, resnet34
from utils.vae_decoder import VAEDecoder
from utils.train_dataset import Gesture2DTrainSet, Sequential2DTestGestureDataSet
from utils.transforms import BaseGroup, GroupColorJitter, GroupNormalize, GroupRandomVerticalFlip, GroupScale, GroupCenterCrop
from utils.transforms import GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupRandomRotation, GroupRandomPerspective
from utils.metrics import accuracy, average_F1, edit_score, overlap_f1
from utils.loss import Loss
from utils import util
from utils.util import AverageMeter, splits_LOSO, splits_LOUO, splits_LOUO_NP, gestures_SU, gestures_NP, gestures_KT
from utils.util import splits_SAR_RARP50, gestures_SAR_RARP50, splits_MultiBypass140, gestures_MultiBypass140
from utils.util import WANDB_API_KEY
#------------------------------------------------------------#

wandb.login(key=WANDB_API_KEY)

INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]
# ACC_TRESHOLDS: List[int] = []  # [85, 92, 96, 98, 99, 99.5, 99.9]


def log(msg, output_folder):
    f_log = open(os.path.join(output_folder, "log.txt"), 'a')
    util.log(f_log, msg)
    f_log.close()


def eval(model, val_loaders, device_gpu, device_cpu, num_class, output_folder, gesture_ids, epoch, upload=False):
    model.eval()
    with torch.no_grad():

        overall_acc = []
        overall_avg_f1 = []
        overall_edit = []
        overall_f1_10 = []
        overall_f1_25 = []
        overall_f1_50 = []

        for val_loader in val_loaders:
            P = np.array([], dtype=np.int64)
            Y = np.array([], dtype=np.int64)

            train_loader_iter = iter(val_loader)
            while True:
                try:
                    (data, target) = next(train_loader_iter)
                except StopIteration:
                    break
                except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
                    print(e)

                # for i, batch in enumerate(val_loader):
                # data, target = batch
                Y = np.append(Y, target.numpy())
                data = data.to(device_gpu)
                output = model(data)

                if len(output.shape) > 2:
                    output = output[:, :, -1]  # consider only final prediction
                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)
                P = np.append(P, predicted.to(device_cpu).numpy())
            acc = accuracy(P, Y)

            mean_avg_f1, avg_precision, avg_recall, avg_f1 = average_F1(P, Y, n_classes=num_class)
            # if upload:
            # avg_precision_table = wandb.Table(data=avg_precision, columns=gestures_SU)
            # wandb.log({"my_custom_plot_id": wandb.plot.line(avg_precision_table, "x", "avg_precision",
            #                                                 title="Custom Y vs X Line Plot")})

            avg_precision_ = np.array(avg_precision)
            avg_recall_ = np.array(avg_recall)
            avg_f1_ = np.array(avg_f1)
            gesture_ids_ = gesture_ids.copy() + ["mean"]
            avg_precision.append(np.mean(avg_precision_[(avg_precision_) != np.array(None)]))
            avg_recall.append(np.mean(avg_recall_[(avg_recall_) != np.array(None)]))
            avg_f1.append(np.mean(avg_f1_[(avg_f1_) != np.array(None)]))
            df = pd.DataFrame(list(zip(gesture_ids_, avg_precision, avg_recall, avg_f1)),
                              columns=['gesture_ids', 'avg_precision', 'avg_recall', 'avg_f1'])
            log(df, output_folder)
            edit = edit_score(P, Y)
            f1_10 = overlap_f1(P, Y, n_classes=num_class, overlap=0.1)
            f1_25 = overlap_f1(P, Y, n_classes=num_class, overlap=0.25)
            f1_50 = overlap_f1(P, Y, n_classes=num_class, overlap=0.5)
            log("Trial {}:\tAcc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1@10 - {:.3f} - F1@25 {:.3f} F1@50 - {:.3f}"
                .format(val_loader.dataset.video_id, acc, mean_avg_f1, edit, f1_10, f1_25, f1_50), output_folder)

            overall_acc.append(acc)
            overall_avg_f1.append(mean_avg_f1)
            overall_edit.append(edit)
            overall_f1_10.append(f1_10)
            overall_f1_25.append(f1_25)
            overall_f1_50.append(f1_50)

        log("Overall: Validation Acc - {:.3f} F1-Macro - {:.3f} Edit - {:.3f} F1@10 - {:.3f} F1@25 - {:.3f} F1@50 - {:.3f}".format(
            np.mean(overall_acc), np.mean(overall_avg_f1), np.mean(overall_edit),
            np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50)
        ), output_folder)

        overall_acc_mean = np.mean(overall_acc)
        if upload:
            wandb.log({'Validation Acc': np.mean(overall_acc), 'F1-Macro': np.mean(overall_avg_f1), 
                       'Edit': np.mean(overall_edit), "F1@10": np.mean(overall_f1_10), "F1@25": np.mean(overall_f1_25),
                       "F1@50": np.mean(overall_f1_50)}, step=epoch)

    return overall_acc_mean


# def main(trial, split=1, upload=False, group=None, args=None):
def main(split=1, upload=False, group=None, args=None):
    print(torch.__version__)
    print(torchvision.__version__)

    # torch.backends.cudnn.enabled = False

    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        print("GPU not found - exit")
        return
    if args is None:
        args = parser.parse_args()

    project_name = args.project_name

    args.eval_batch_size = 2 * args.batch_size
    args.split = split

    upload = upload and (not args.test) # if test, do not upload 
    is_test = args.test 

    # device_gpu = torch.device("cuda")
    device_gpu = torch.device(f"cuda:{args.gpu_id}")
    device_cpu = torch.device("cpu")

    checkpoint = None
    if args.resume_exp:
        output_folder = args.resume_exp
    else:
        # output_folder = os.path.join(args.out, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
        #                              args.eval_scheme, str(args.split), datetime.datetime.now().strftime("%H%M"))
        if is_test:
            output_folder = os.path.join(args.out, args.dataset,
                                         args.arch, "test-" + args.exp + "-test",
                                         args.eval_scheme, str(args.split))

            os.makedirs(output_folder, exist_ok=True)

        else:
            if args.dataset in ['JIGSAWS', 'SAR_RARP50', 'MultiBypass140']:
                output_folder = os.path.join(args.out, 
                                             args.dataset,
                                             args.arch,
                                             args.eval_scheme, 
                                             str(args.split))
            else: 
                raise NotImplementedError()
            

            if os.path.exists(output_folder):
                print("Output folder already exists. Do you want to delete it? (y/n)")
                user_input = input()
                if user_input.lower() == 'y':
                    shutil.rmtree(output_folder)
                else:
                    print("Please delete the existing folder before running the code.")
                    return
            os.makedirs(output_folder, exist_ok=False)

    checkpoint_file = os.path.join(output_folder, "checkpoint" + ".pth.tar")

    if args.resume_exp:
        checkpoint = torch.load(checkpoint_file)
        args_checkpoint = checkpoint['args']
        for arg in args_checkpoint:
            setattr(args, arg, args_checkpoint[arg])
        log("====================================================================", output_folder)
        log("Resuming experiment...", output_folder)
        log("====================================================================", output_folder)
    else:
        # if args.dataset == "JIGSAWS":
        #     if len([t for t in string.Formatter().parse(args.data_path)]) > 1:
        #         args.data_path = args.data_path.format(args.task)
        #     if len([t for t in string.Formatter().parse(args.video_lists_dir)]) > 1:
        #         args.video_lists_dir = args.video_lists_dir.format(args.task)
        #     if len([t for t in string.Formatter().parse(args.transcriptions_dir)]) > 1:
        #         args.transcriptions_dir = args.transcriptions_dir.format(args.task)

        log("Used parameters...", output_folder)
        for arg in sorted(vars(args)):
            log("\t" + str(arg) + " : " + str(getattr(args, arg)), output_folder)

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if checkpoint:
        torch.set_rng_state(checkpoint['rng'])

    # ===== prepare model =====
    gesture_ids = get_gestures(args.dataset, args.task)

    num_class = len(gesture_ids)

    # check num of classes
    if args.num_classes is not None:
        assert args.num_classes == num_class, f'Number of classes defined in \'util.py\' is {num_class} and not {args.num_classes} as defined in the arguments (--num_classes)!'
    # check epoch size and number of samples per class and batch size
    # if args.epoch_size is not None:
    #     assert args.epoch_size % args.batch_size == 0, f'Epoch size must be divisible by the batch size!'
    #     assert args.epoch_size == (args.number_of_samples_per_class * num_class), f'Epoch size (--epoch_size={args.epoch_size}) must be equal to the number of samples per class (--number_of_samples_per_class={args.number_of_samples_per_class}) times the number of classes (--num_classes={num_class})!'

    if args.word_embdding_weight:
        assert args.label_embedding_path

        label_embedding = torch.load(args.label_embedding_path)
        embedding_shape = label_embedding.shape[-1]

    else:
        embedding_shape = 0

    model = get_model(args.arch, num_classes=num_class,
                      add_layer_param_num=args.additional_param_num,
                      add_certainty_pred=args.certainty_weight,
                      input_shape=args.input_size if args.decoder_weight else 0,
                      embedding_shape=embedding_shape,
                      vae_intermediate_size=args.vae_intermediate_size)

    if checkpoint:
        # load model weights
        model.load_state_dict(checkpoint['model_weights'])

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log("param count: {}".format(param_count), output_folder)
    log("trainable params: {}".format(trainable_params), output_folder)

    # inp_for_test = torch.rand(1,3, 16, 224, 224).to(device_cpu)
    # # model(inp_for_test)
    #
    # # # Count the number of FLOPs
    # count_ops(model, inp_for_test)

    criterion = Loss(decoder_weight=args.decoder_weight,
                     certainty_weight=args.certainty_weight,
                     word_embdding_weight=args.word_embdding_weight,
                     class_criterion_weight=args.class_criterion_weight,
                     word_embdding_loss_param={"positive_aggregator": args.positive_aggregator,
                                               "margin": args.margin,
                                               "label_embedding": label_embedding.to(
                                                   device_gpu) if args.word_embdding_weight else None},
                     vae_loss_param={"x_sigma2": args.x_sigma2}
                     )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if checkpoint:
        # load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device_gpu)
    scheduler = None
    if args.use_scheduler:
        last_epoch = -1
        if checkpoint:
            last_epoch = checkpoint['epoch']
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2, last_epoch=last_epoch)

    # ===== load data =====

    if args.dataset == "JIGSAWS":
        lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
    elif args.dataset in ['SAR_RARP50', 'MultiBypass140']:
        lists_dir = args.video_lists_dir
    
    splits = get_splits(args.dataset, args.eval_scheme, args.task)
    
    if args.dataset in ['JIGSAWS', 'SAR_RARP50']:
        train_lists, val_list = train_val_split(splits, args.split)
        train_lists = list(map(lambda x: os.path.join(lists_dir, x), train_lists))
        val_lists   = list(map(lambda x: os.path.join(lists_dir, x), val_list))
    elif args.dataset == "MultiBypass140":
        train_lists, val_list = train_val_split(splits, args.split, MB140=True)
        train_lists = list(map(lambda x: os.path.join(lists_dir, x), train_lists.split()))
        val_lists   = list(map(lambda x: os.path.join(lists_dir, x), val_list.split()))
    
    # TODO Add HERE test_lists for other datasets
    log("Splits in train set :" + str(train_lists), output_folder)
    log("Splits in valid set :" + str(val_lists), output_folder)

    normalize = GroupNormalize(INPUT_MEAN, INPUT_STD)
    train_augmentation = get_augmentation(args.input_size, crop_corners=args.corner_cropping,
                                          do_horizontal_flip=args.do_horizontal_flip,
                                          do_vertical_flip=args.do_vertical_flip,
                                          perspective_distortion=args.perspective_distortion,
                                          degrees=args.degrees,
                                          do_color_jitter=args.do_color_jitter)
    train_set = Gesture2DTrainSet(args.dataset, args.data_path, train_lists, args.transcriptions_dir, gesture_ids,
                                  image_tmpl=args.image_tmpl, video_suffix=args.video_suffix,
                                  transform=train_augmentation, normalize=normalize, resize=args.input_size, debag=False,
                                  number_of_samples_per_class=args.number_of_samples_per_class, preload=args.preload)

    def no_none_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def init_train_loader_worker(worker_id):
        np.random.seed(int((torch.initial_seed() + worker_id) % (2 ** 32)))  # account for randomness

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, worker_init_fn=init_train_loader_worker,
                                               collate_fn=no_none_collate)
    log("Training set: will sample {} gesture snippets per pass".format(train_loader.dataset.__len__()), output_folder)

    val_augmentation = torchvision.transforms.Compose([GroupScale(args.input_size), GroupCenterCrop(args.input_size)])
    
    val_videos = list()
    for list_file in val_lists:
        # format should be video_id, frame_count
        val_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
    val_loaders = list()

    if args.test:
        val_videos = val_videos[:2]

    for video in val_videos:
        data_set = Sequential2DTestGestureDataSet(dataset=args.dataset, root_path=args.data_path, sar_rarp50_sub_dir='train', video_id=video[0], frame_count=video[1],
                                                  transcriptions_dir=args.transcriptions_dir, gesture_ids=gesture_ids,
                                                  snippet_length=args.snippet_length,
                                                  sampling_step=args.val_sampling_step,
                                                  image_tmpl=args.image_tmpl,
                                                  video_suffix=args.video_suffix,
                                                  normalize=normalize, resize=args.input_size,
                                                  transform=val_augmentation)  # augmentation are off
        val_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
                                                       shuffle=False, num_workers=args.workers,
                                                       collate_fn=no_none_collate))

    log("Validation set: ", output_folder)
    for val_loader in val_loaders:
        log("{} ({})".format(val_loader.dataset.video_id, val_loader.dataset.__len__()), output_folder)

    if upload:
        configuration = vars(args)
        configuration["param count"] = param_count
        configuration["trainable param count"] = trainable_params

        run_id = checkpoint["run_id"] if checkpoint else None
        if group is None:
            group = args.exp
        elif len([t for t in string.Formatter().parse(group)]) == 3:
            group = group.format(args.arch, args.dataset, args.weight_decay)
        elif len([t for t in string.Formatter().parse(group)]) == 2:
            group = group.format(args.arch, args.dataset)

        wandb.init(project=project_name, config=configuration,
                   id=run_id, resume=checkpoint, group=group,
                   name=f"{args.arch}_{args.eval_scheme}_{split}", reinit=True,
                   dir=os.path.dirname(os.path.abspath(__file__)))

    # ===== train model =====

    log("Start training...", output_folder)

    model = model.to(device_gpu)

    start_epoch = 0
    if checkpoint:
        start_epoch = checkpoint['epoch']

    max_acc_val = 0
    max_acc_train = 0

    # acc_tresholds = {key / 100: False for key in sorted(ACC_TRESHOLDS)}
    # finished_acc_tresholds = False

    for epoch in range(start_epoch, args.epochs):
        if epoch > start_epoch:
            log(f"Training set: sampling {train_loader.dataset.__len__()} gesture snippets for epoch number {epoch}", output_folder)
            train_set.randomize() # resample the dataset for each epoch
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        model.train()

        with tqdm.tqdm(desc=f'{"Epoch"} ({epoch}) {"progress"}', total=int(len(train_loader))) as pbar:

            # for batch_i, (data, target) in enumerate(train_loader):

            train_loader_iter = iter(train_loader)
            while True:
                try:
                    pbar.update(1)
                    (data, target) = next(train_loader_iter)
                except StopIteration:
                    break
                except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
                    print(e)

                # for batch in train_loader:
                optimizer.zero_grad()
                # data, target = batch
                data = Variable(data.to(device_gpu))
                target = Variable(target.to(device_gpu), requires_grad=False)
                batch_size = target.size(0)
                # target = target.to(device_gpu, dtype=torch.int64)

                output = model(data)

                loss = criterion(output, target, img=data)

                # target = target.to(dtype=torch.float)

                loss = torch.mean(loss)

                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), batch_size)

                if isinstance(output, Sequence):
                    output = output[0]
                elif isinstance(output, Dict):
                    output = output["class_prob"]

                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)
                # _, predicted = torch.max(output, 1)

                acc = (predicted == target).sum().item() / batch_size

                train_acc.update(acc, batch_size)

            pbar.close()
            if scheduler is not None:
                scheduler.step()

        log("Epoch {}: Train loss: {train_loss.avg:.4f} Train acc: {train_acc.avg:.3f}"
            .format(epoch, train_loss=train_loss, train_acc=train_acc), output_folder)
        if upload:
            wandb.log({'train accuracy': train_acc.avg, 'loss': train_loss.avg}, step=epoch)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            log("Start testing...", output_folder)

            overall_acc_mean = eval(model, val_loaders, device_gpu, device_cpu, num_class, output_folder, gesture_ids, epoch, upload=upload)
            max_acc_val = overall_acc_mean # Gabriel Gozal - No Stopping criteria
            # if overall_acc_mean > max_acc_val and not is_test:
            #     max_acc_val = overall_acc_mean
            #     model_file = os.path.join(output_folder, "best_" + f"{args.split}" + ".pth")
            #     torch.save(model.state_dict(), model_file)
            #     log("Saved best model to " + model_file, output_folder)

        # if train_acc.avg > max_acc_train and not is_test:
        #     max_acc_train = train_acc.avg
        #     model_file = os.path.join(output_folder, "best_train_" + f"{args.split}" + ".pth")
        #     torch.save(model.state_dict(), model_file)
        #     log("Saved best train model to " + model_file, output_folder)

        # for tresh, achived in acc_tresholds.items():
        #     if achived:
        #         continue

        #     if train_acc.avg >= tresh:
        #         model_file = os.path.join(output_folder,
        #                                   f"acc_{tresh * 100}_epoch_{epoch}_split_" + f"{args.split}" + ".pth")
        #         torch.save(model.state_dict(), model_file)
        #         log(f"Saved model with {tresh * 100} % acc to " + model_file, output_folder)

        #         acc_tresholds[tresh] = True
        #         break

        # early stopping when all the trsholds are achived
        # if all([achived for tresh, achived in acc_tresholds.items()]):
        #     finished_acc_tresholds = True

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1 and not is_test:
            # ===== save model =====
            model_file = os.path.join(output_folder, "model_" + str(epoch) + ".pth")
            torch.save(model.state_dict(), model_file)
            log("Saved model to " + model_file, output_folder)

        # ===== save checkpoint =====
        current_state = {'epoch': epoch + 1,
                         'model_weights': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         'rng': torch.get_rng_state(),
                         'args': args_dict
                         }
        if upload:
            current_state["run_id"] = wandb.run.id

        if not is_test:
            torch.save(current_state, checkpoint_file)

        # if finished_acc_tresholds:
        #     print("acc tresholds finished")
        #     break

    wandb.summary["validation accuracy"] = max_acc_val
    return max_acc_val


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
    elif dataset == "SAR_RARP50":
        splits = splits_SAR_RARP50
    elif dataset == "MultiBypass140":
        splits = splits_MultiBypass140
    else:
        raise NotImplementedError()

    return splits


def train_val_split(splits, val_split, MB140=False):
    if isinstance(val_split, int) and (not MB140):
        assert (val_split >= 0 and val_split < len(splits))
        train_lists = splits[0:val_split] + splits[val_split + 1:]
        val_list = splits[val_split:val_split + 1]

        if isinstance(train_lists[0], list):
            train_lists = [item for train_split in train_lists for item in train_split]
        if isinstance(val_list[0], list):
            val_list = [item for val_split in val_list for item in val_split]
    elif MB140:
        assert (val_split >= 0 and val_split < len(splits['train']))
        train_lists = splits['train'][val_split]
        val_list    = splits['val'][val_split]
    else:
        assert isinstance(val_split, List)
        assert all((s in splits) for s in val_split)
        train_lists = [s for s in splits if s not in val_split]
        val_list = [s for s in splits if s in val_split]

    return train_lists, val_list


def get_augmentation(input_size, crop_corners=True,
                     do_horizontal_flip=True,
                     do_vertical_flip=False,
                     perspective_distortion=0,
                     degrees=0,
                     do_color_jitter=False):
    augmenations = [GroupMultiScaleCrop(input_size, [1, .875, .75, .66],
                                        fix_crop=crop_corners,
                                        more_fix_crop=crop_corners)]

    if do_horizontal_flip:
        augmenations.append(GroupRandomHorizontalFlip(is_flow=False))

    if do_vertical_flip:
        augmenations.append(GroupRandomVerticalFlip())

    if perspective_distortion:
        augmenations.append(GroupRandomPerspective(distortion_scale=perspective_distortion, p=0.33))

    if degrees:
        augmenations.append(GroupRandomRotation(degrees=degrees))

    if do_color_jitter:
        augmenations.append(GroupColorJitter(brightness=0.2, contrast=0.1))

    return torchvision.transforms.Compose(augmenations)


def get_gestures(dataset, task=None):
    if dataset == "JIGSAWS":
        if task == "Suturing":
            gesture_ids = gestures_SU
        elif task == "Needle_Passing":
            gesture_ids = gestures_NP
        elif task == "Knot_Tying":
            gesture_ids = gestures_KT
    elif dataset == "SAR_RARP50":
        gesture_ids = gestures_SAR_RARP50
    elif dataset == "MultiBypass140":
        gesture_ids = gestures_MultiBypass140
    else:
        raise NotImplementedError()

    return gesture_ids


def get_model(arch, num_classes=1000, remove_end=False, pretrained=True, progress=True,
              add_layer_param_num=0, input_shape=0, add_certainty_pred=False, legacy=False,
              embedding_shape=0, vae_intermediate_size=None):
    """create a model based on the pararamters given and return it

    :param arch: base architecture of the model
    :param num_classes: number of classes for the model to predict, defaults to 1000
    :param remove_end: whether to remove the last layer of the model.
    if add_layer_pararm_num != 0, the activation and the dropout are removed additionally, defaults to False
    :param pretrained: whether to load pretrained weights from imagenet , defaults to True
    :param progress: relevant if if pretrain is True, show the weight download process, defaults to True
    :param add_layer_param_num: number of parameters in additional layer.
    if 0 then no additional layer will be created. an activation function and dropout are added as well,
    defaults to 0
    """
    if legacy:
        return get_model_legacy(arch, num_classes, remove_end, pretrained,
                                progress, add_layer_param_num)

    if arch in ["2D-ResNet-18", "2D-ResNet-34"]:
        if arch == "2D-ResNet-18":
            model = resnet18(pretrained=pretrained, progress=progress, num_classes=add_layer_param_num)
        else:
            model = resnet34(pretrained=pretrained, progress=progress, num_classes=add_layer_param_num)

        if add_layer_param_num != 0:
            penultimate_shape = add_layer_param_num
        else:
            penultimate_shape = 512

        def _two_head_forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return self.added_fc(x), x

    elif arch in ["2D-EfficientNetV2-s", "2D-EfficientNetV2-m", "2D-EfficientNetV2-l"]:
        if arch == "2D-EfficientNetV2-s":
            model = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=add_layer_param_num)
        elif arch == "2D-EfficientNetV2-m":
            model = timm.create_model("tf_efficientnetv2_m", pretrained=pretrained, num_classes=add_layer_param_num)
        elif arch == "2D-EfficientNetV2-l":
            model = timm.create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=add_layer_param_num)
        else:
            raise NotImplementedError()

        if add_layer_param_num != 0:
            penultimate_shape = add_layer_param_num
        else:
            penultimate_shape = 1280

        def _two_head_forward(self, x):
            x = self.forward_features(x)
            x = self.global_pool(x)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)

            x = self.classifier(x)

            return self.added_fc(x), x
    else:
        raise NotImplementedError(f"required architecture: {arch} - not implemented")

    if input_shape:
        if vae_intermediate_size is None:
            vae_intermediate_size = penultimate_shape

        model.decoder = VAEDecoder(intermediate_image_shape=input_shape,
                                   input_size=penultimate_shape,
                                   intermediate_size=vae_intermediate_size)
    if add_certainty_pred:
        model.certainty_pred = nn.Sequential(nn.Linear(penultimate_shape, 1),
                                             nn.Sigmoid())
    if embedding_shape:
        model.embedding_creator = nn.Linear(penultimate_shape, embedding_shape)

    if remove_end:
        model.added_fc = Identity()
    elif add_layer_param_num:
        model.added_fc = nn.Sequential(nn.Mish(), nn.Dropout(0.5), nn.Linear(penultimate_shape, num_classes))
    else:
        model.added_fc = nn.Linear(penultimate_shape, num_classes)

    if not add_layer_param_num:
        if "ResNet" in arch:
            model.fc = Identity()
        elif "EfficientNet" in arch:
            model.classifier = Identity()

    model._two_head_forward = MethodType(_two_head_forward, model)

    def forward(self, x):
        output, penultimate_output = self._two_head_forward(x)
        res = output

        if self.training:
            res = {"class_prob": output}

            if hasattr(self, "decoder"):
                res["decoded_image"] = self.decoder(penultimate_output)

            if hasattr(self, "certainty_pred"):
                certainty_pred = self.certainty_pred(penultimate_output)
                res["certainty_pred"] = certainty_pred

            if hasattr(self, "embedding_creator"):
                res["embedding"] = self.embedding_creator(penultimate_output)

        return res

    model.forward = MethodType(forward, model)

    return model


def get_model_legacy(arch, num_classes=1000, remove_end=False, pretrained=True, progress=True,
                     add_layer_param_num=0):
    """create a model based on the pararamters given and return it

    :param arch: base architecture of the model
    :param num_classes: number of classes for the model to predict, defaults to 1000
    :param remove_end: whether to remove the last layer of the model.
    if add_layer_pararm_num != 0, the activation and the dropout are removed additionally, defaults to False
    :param pretrained: whether to load pretrained weights from imagenet , defaults to True
    :param progress: relevant if if pretrain is True, show the weight download process, defaults to True
    :param add_layer_param_num: number of parameters in additional layer.
    if 0 then no additional layer will be created. an activation function and dropout are added as well,
    defaults to 0
    """
    if add_layer_param_num != 0:
        assert (add_layer_param_num > 0)
        out_param_num = add_layer_param_num
    else:
        out_param_num = num_classes

    if arch in ["2D-ResNet-18", "2D-ResNet-34"]:
        if arch == "2D-ResNet-18":
            model = resnet18(pretrained=pretrained, progress=progress, num_classes=out_param_num)
        else:
            model = resnet34(pretrained=pretrained, progress=progress, num_classes=out_param_num)

    elif arch in ["2D-EfficientNetV2-s", "2D-EfficientNetV2-m", "2D-EfficientNetV2-l"]:
        if arch == "2D-EfficientNetV2-s":
            model = timm.create_model("tf_efficientnetv2_s", pretrained=pretrained, num_classes=out_param_num)
        elif arch == "2D-EfficientNetV2-m":
            model = timm.create_model("tf_efficientnetv2_m", pretrained=pretrained, num_classes=out_param_num)
        else:
            model = timm.create_model("tf_efficientnetv2_l", pretrained=pretrained, num_classes=out_param_num)

    else:
        raise NotImplementedError("required architecture not implemented")

    if remove_end and add_layer_param_num == 0:
        if "ResNet" in arch:
            model.fc = Identity()
        elif "EfficientNet" in arch:
            model.classifier = Identity()

    if add_layer_param_num != 0 and not remove_end:
        model = nn.Sequential(*[model, nn.Mish(), nn.Dropout(0.5), nn.Linear(out_param_num, num_classes)])
    else:
        # model = nn.Sequential(*[model])
        pass

    return model


def load_model(weights_path, arch, add_layer_param_num=0,
               remove_linear=True, add_certainty_pred=False,
               num_classes=1000, decoder_input_size=0, vae_intermediate_size=0):
    model = get_model(arch, remove_end=remove_linear,
                      pretrained=False, add_layer_param_num=add_layer_param_num,
                      add_certainty_pred=add_certainty_pred, num_classes=num_classes,
                      input_shape=decoder_input_size,
                      vae_intermediate_size=vae_intermediate_size)

    incompatible_keys = model.load_state_dict(torch.load(weights_path), strict=False)

    if len(incompatible_keys[0]) != 0:  # support for legacy code
        model = get_model(arch, remove_end=remove_linear, pretrained=False,
                          add_layer_param_num=add_layer_param_num, legacy=True)
        incompatible_keys = model.load_state_dict(torch.load(weights_path), strict=False)

        if len(incompatible_keys[0]) != 0:
            model = nn.Sequential(model)
            incompatible_keys = model.load_state_dict(torch.load(weights_path), strict=False)

    if len(incompatible_keys[0]) != 0:
        raise RuntimeError("Missing weights from weight path", weights_path)

    return model


def run_full_LOUO(group_name=None):
    args = parser.parse_args()

    if args.dataset == "MultiBypass140":
        user_num = len(splits_MultiBypass140['train'])
    elif args.dataset == "JIGSAWS":
        user_num = len(splits_LOUO)
    elif args.dataset == "SAR_RARP50":
        user_num = len(splits_SAR_RARP50)
    else: # 'VTS'
        raise NotImplementedError(f"{args.dataset} not implemented")
    # if group_name is None:
    # group_name = f"{args.arch} cross validation {args.dataset}"

    for i in range(user_num):
        # main(0, split=i, upload=True, group=group_name)
        main(split=i, upload=True, group=group_name)

def run_full_LOSO(group_name=None):
    args = parser.parse_args()

    if args.dataset in ['VTS', 'MultiBypass140', 'SAR_RARP50']:
        raise NotImplementedError(f"{args.dataset} not implemented")
    elif args.dataset == "JIGSAWS":
        supertrial_num = len(splits_LOSO)
    
    # if group_name is None:
    # group_name = f"{args.arch} cross validation {args.dataset}"

    for i in range(supertrial_num):
        # main(0, split=i, upload=True, group=group_name)
        main(split=i, upload=True, group=group_name)

def get_k_folds_splits(k=5, shuffle=True, args=None):
    if not args:
        args = parser.parse_args()

    if args.dataset == 'JIGSAWS':
        users = splits_LOUO
    elif args.dataset == 'SAR_RARP50':
        users = splits_SAR_RARP50
    else: 
        raise NotImplementedError()

    if shuffle:
        kfolds = KFold(n_splits=k, shuffle=shuffle, random_state=args.seed)
    else:
        kfolds = KFold(n_splits=k, shuffle=shuffle)

    for train, test in kfolds.split(users):
        split = [users[i] for i in test]
        yield split


def run_k_folds_validation(k=5, shuffle=True, group_name=None):
    args = parser.parse_args()

    for split in get_k_folds_splits(k, shuffle=shuffle):
        # main(0, split=split, upload=True, group=group_name)
        main(split=split, upload=True, group=group_name)


def run_single_split(split_idx=0):
    args = parser.parse_args()
    group_name = f"{args.arch} {args.dataset}"
    # main(0, split=split_idx, upload=True, group=group_name)
    main(split=split_idx, upload=True, group=group_name)


def run():
    args = parser.parse_args()

    if args.dataset in ['VTS']:
        raise NotImplementedError(f"{args.dataset} not implemented")

    elif args.dataset in ['JIGSAWS', 'SAR_RARP50', 'MultiBypass140']:
        if args.split_num is not None:
            # main(0, split=args.split_num, upload=True)
            main(split=args.split_num, upload=True)
        elif args.eval_scheme == "LOUO" or args.dataset in ['SAR_RARP50', 'MultiBypass140']:
            run_full_LOUO()
        elif args.eval_scheme == "LOSO":
            run_full_LOSO()
    # elif args.dataset == 'SAR_RARP50':
    #     if args.split_num is not None:
    #         # main(0, split=args.split_num, upload=True)
    #         main(split=args.split_num, upload=True)
    #     else:
    #         run_k_folds_validation(shuffle=False)
    # elif args.dataset == "MultiBypass140":

    #     if args.split_num is not None:

    #         main(0, split=list(get_k_folds_splits(k=5, shuffle=False))
    #              [args.split_num], upload=True)
    #     else:
    #         run_k_folds_validation(shuffle=False)

    else:  # VTS
        main(upload=True)


if __name__ == '__main__':
    run()

    # run_full_LOUO(group_name="{} cross validation {} added augmentations 2")

    # run_k_folds_validation(shuffle=False)

    # run_single_split(1)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # Experiment_name ="Exp_name"
    # study = optuna.create_study(study_name="Exp_name", load_if_exists=True, storage=os.path.join("sqlite:///", Experiment_name))
    # study.optimize(main, n_trials=10)
