# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology
#----------------- Python Libraries Imports -----------------#
import os.path
import datetime
import string
import random

# Third-party imports
import numpy as np
import pandas as pd
import PIL
import torch
import torchvision
from torch.autograd import Variable
import tqdm
import wandb
#------------------ Bounded Future Imports ------------------#
import utils_2D.util
from utils_2D.train_opts_2D import parser
from utils_2D.resnet2D import resnet18
from utils_2D.efficientnetV2 import EfficientnetV2
from utils_2D.dataset import Gesture2dTrainSet, Sequential2DTestGestureDataSet
from utils_2D.transforms import GroupNormalize, GroupScale, GroupCenterCrop
from utils_2D.metrics import accuracy, average_F1, edit_score, overlap_f1
from utils_2D.util import AverageMeter
from utils_2D.util import splits_VTS, gestures_VTS, splits_JIGSAWS, gestures_JIGSAWS, splits_SAR_RARP50, gestures_SAR_RARP50, splits_MultiBypass140, steps_MultiBypass130, phases_MultiBypass130
from utils_2D.util import WANDB_API_KEY
#------------------------------------------------------------#

args = parser.parse_args()

gesture_ids = (gestures_VTS if args.dataset == "VTS" else 
               gestures_JIGSAWS if args.dataset == "JIGSAWS" else
               gestures_SAR_RARP50 if args.dataset == "SAR_RARP50" else 
               steps_MultiBypass130 if args.dataset == "MultiBypass140" and args.task == 'steps' else
               phases_MultiBypass130 if args.dataset == "MultiBypass140" and args.task == 'phases' else None)
folds_folder = (os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', args.dataset, 'folds'))
num_of_splits = (5 if args.dataset in ["VTS", "MultiBypass140", "SAR_RARP50"] else
                 8 if args.dataset == "JIGSAWS" else None)
args.transcriptions_dir = (args.transcriptions_dir if args.dataset != "MultiBypass140" else
                           os.path.join(args.transcriptions_dir, args.task))
if args.wandb:
    wandb.login(key=WANDB_API_KEY) 
#---------------------------------VTS---------------------------------#
def read_VTS_data(folds_folder,split_num):
    list_of_train_examples = []
    number_of_folds = 0
    for file in os.listdir(folds_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "fold" in filename:
            number_of_folds = number_of_folds + 1

    for file in os.listdir(folds_folder):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "fold" in filename:
            if str(split_num) in filename:
                file_ptr = open(os.path.join(folds_folder, filename), 'r')
                list_of_test_examples = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                random.shuffle(list_of_test_examples)
            elif str((split_num + 1) % number_of_folds) in filename:
                file_ptr = open(os.path.join(folds_folder, filename), 'r')
                list_of_examples = file_ptr.read().split('\n')[:-1]
                list_of_valid_examples = list_of_examples[0:12]
                random.shuffle(list_of_valid_examples)
                list_of_train_examples = list_of_train_examples + list_of_examples[12:]

                file_ptr.close()
            else:
                file_ptr = open(os.path.join(folds_folder, filename), 'r')
                list_of_train_examples = list_of_train_examples + file_ptr.read().split('\n')[:-1]
                file_ptr.close()
            continue
        else:
            continue

    random.shuffle(list_of_train_examples)
    return list_of_train_examples, list_of_valid_examples, list_of_test_examples
#-------------------------------JIGSAWS-------------------------------#
def read_JIGSAWS_data(folds_folder,split_num):
    list_of_train_examples = []
    number_of_folds = 0
    for file in sorted(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "valid" in filename:
            number_of_folds = number_of_folds + 1

    for file in sorted(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "valid" in filename:
            file_path = os.path.join(folds_folder, filename)
            #-------------------------------------------#
            # check if the file exists and is not empty #
            #-------------------------------------------#
            if not os.path.exists(file_path):
                print(f"The file {file_path} does not exist.")
                raise FileNotFoundError
            elif os.path.getsize(file_path) == 0:
                print(f"The file {file_path} is empty.")
                raise EOFError
            else:
                with open(file_path, 'r') as file_ptr:
                        contents = file_ptr.read()
                if '\n' not in contents:
                    print(f"The file {file_path} does not contain any newline characters.")
                else:
                    files_to_sort = contents.split('\n')[:-1]
            #-------------------------------------------#
                #----- Validation set -----# NOTICE: There is no Validation set in JIGSAWS dataset
                if str(split_num) in filename:
                    list_of_valid_examples = files_to_sort
                    if not list_of_valid_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
                    else:
                        random.shuffle(list_of_valid_examples)
                #----- Test set -----# Using the same data for test and validation since there is no validation set in JIGSAWS dataset
                if str(split_num) in filename:
                        list_of_test_examples = files_to_sort
                        if not list_of_test_examples:
                            print(f"The file {file_path} only contains empty lines or ends with a newline.")
                            raise EOFError
                        else:
                            random.shuffle(list_of_test_examples)
                #----- Training set -----#
                else:
                    list_of_train_examples = list_of_train_examples + files_to_sort
                    if not list_of_train_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
            continue
        else:
            continue

    random.shuffle(list_of_train_examples)
    return list_of_train_examples, list_of_valid_examples, list_of_test_examples
#----------------------------MultiBypass140---------------------------#
def read_MultiBypass140_data(folds_folder,split_num):
    list_of_train_examples = []
    number_of_folds = 0
    for file in sorted(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "val" in filename:
            number_of_folds = number_of_folds + 1

    for file in sorted(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and str(split_num) in filename:
            file_path = os.path.join(folds_folder, filename)
            #-------------------------------------------#
            # check if the file exists and is not empty #
            #-------------------------------------------#
            if not os.path.exists(file_path):
                print(f"The file {file_path} does not exist.")
                raise FileNotFoundError
            elif os.path.getsize(file_path) == 0:
                print(f"The file {file_path} is empty.")
                raise EOFError
            else:
                with open(file_path, 'r') as file_ptr:
                    contents = file_ptr.read()
                if '\n' not in contents:
                    print(f"The file {file_path} does not contain any newline characters.")
                else:
                    files_to_sort = contents.split('\n')[:-1]
            #-------------------------------------------#
            if "val" in filename:
                #----- Validation set -----# 
                list_of_valid_examples = files_to_sort
                if not list_of_valid_examples:
                    print(f"The file {file_path} only contains empty lines or ends with a newline.")
                    raise EOFError
                else:
                    random.shuffle(list_of_valid_examples)
            elif "train" in filename:
                #----- Training set -----#
                list_of_train_examples = files_to_sort
                if not list_of_train_examples:
                    print(f"The file {file_path} only contains empty lines or ends with a newline.")
                    raise EOFError
                else:
                    random.shuffle(list_of_train_examples)
            elif "test" in filename:
                #----- Test set -----# 
                list_of_test_examples = files_to_sort
                if not list_of_test_examples:
                    print(f"The file {file_path} only contains empty lines or ends with a newline.")
                    raise EOFError
                else:
                    random.shuffle(list_of_test_examples)
    return list_of_train_examples, list_of_valid_examples, list_of_test_examples
#-----------------------------SAR-RARP50------------------------------#
def read_SAR_RARP50_data(folds_folder,split_num):
    list_of_train_examples = []
    number_of_folds = 0
    for file in sorted(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt") and "valid" in filename:
            number_of_folds = number_of_folds + 1

    for file in sorted(os.listdir(folds_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            file_path = os.path.join(folds_folder, filename)
            #-------------------------------------------#
            # check if the file exists and is not empty #
            #-------------------------------------------#
            if not os.path.exists(file_path):
                print(f"The file {file_path} does not exist.")
                raise FileNotFoundError
            elif os.path.getsize(file_path) == 0:
                print(f"The file {file_path} is empty.")
                raise EOFError
            else:
                with open(file_path, 'r') as file_ptr:
                        contents = file_ptr.read()
                if '\n' not in contents:
                    print(f"The file {file_path} does not contain any newline characters.")
                else:
                    files_to_sort = contents.split('\n')[:-1]
            #-------------------------------------------#
            if "valid" in filename:
                #----- Validation set -----# 
                if str(split_num) in filename:
                    list_of_valid_examples = files_to_sort
                    if not list_of_valid_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
                    else:
                        random.shuffle(list_of_valid_examples)
                #----- Training set -----#
                else:
                    list_of_train_examples = list_of_train_examples + files_to_sort
                    if not list_of_train_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
            elif "test" in filename:
                #----- Test set -----# 
                list_of_test_examples = files_to_sort
                if not list_of_test_examples:
                    print(f"The file {file_path} only contains empty lines or ends with a newline.")
                    raise EOFError
                else:
                    random.shuffle(list_of_test_examples)
        else:
            continue

    random.shuffle(list_of_train_examples)
    return list_of_train_examples, list_of_valid_examples, list_of_test_examples
#---------------------------------------------------------------------# 


def log(msg,output_folder):
    f_log = open(os.path.join(output_folder, "log.txt"), 'a')
    utils_2D.util.log(f_log, msg)
    f_log.close()

def eval(model,val_loaders,device_gpu,device_cpu,num_class,output_folder,gesture_ids,epoch,upload=False):
    results_per_vedo= []
    all_precisions = []
    all_recalls = []
    all_f1s =[]
    model.eval()
    with torch.no_grad():

        overall_acc = []
        overall_avg_f1 = []
        overall_edit = []
        overall_f1_10 = []
        overall_f1_25 = []
        overall_f1_50 = []
        for video_num, val_loader in enumerate(val_loaders):
            P = np.array([], dtype=np.int64)
            Y = np.array([], dtype=np.int64)
            for i, batch in enumerate(val_loader):
                data, target = batch
                Y = np.append(Y, target.numpy())
                data = data.to(device_gpu)
                output = model(data)
                if model.arch == "EfficientnetV2":
                    output = output[0]

                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)

                P = np.append(P, predicted.to(device_cpu).numpy())

            acc = accuracy(P, Y)
            mean_avg_f1, avg_precision, avg_recall, avg_f1 = average_F1(P, Y, n_classes=num_class)
            all_precisions.append(avg_precision)
            all_recalls.append(avg_recall)
            all_f1s.append(avg_f1)

            avg_precision_ = np.array(avg_precision)
            avg_recall_ = np.array(avg_recall)
            avg_f1_ = np.array(avg_f1)
            avg_precision.append(np.mean(avg_precision_[(avg_precision_) != np.array(None)]))
            avg_recall.append(np.mean(avg_recall_[(avg_recall_) != np.array(None)]))
            avg_f1.append(np.mean(avg_f1_[(avg_f1_) != np.array(None)]))
            edit = edit_score(P, Y)
            f1_10 = overlap_f1(P, Y, n_classes=num_class, overlap=0.1)
            f1_25 = overlap_f1(P, Y, n_classes=num_class, overlap=0.25)
            f1_50 = overlap_f1(P, Y, n_classes=num_class, overlap=0.5)
            log("Trial {}:\tAcc\t{:.3f}\tAvg_F1\t{:.3f}\tEdit\t{:.3f}\tF1_10\t{:.3f}\tF1_25\t{:.3f}\tF1_50\t{:.3f}"
                .format(val_loader.dataset.video_name, acc, mean_avg_f1, edit, f1_10, f1_25, f1_50), output_folder)
            results_per_vedo.append([val_loader.dataset.video_name, acc, mean_avg_f1, edit, f1_10, f1_25, f1_50])

            overall_acc.append(acc)
            overall_avg_f1.append(mean_avg_f1)
            overall_edit.append(edit)
            overall_f1_10.append(f1_10)
            overall_f1_25.append(f1_25)
            overall_f1_50.append(f1_50)

        gesture_ids_ = gesture_ids.copy() + ["mean"]
        all_precisions = np.array(all_precisions).mean(0)
        all_recalls =  np.array(all_recalls).mean(0)
        all_f1s = np.array(all_f1s).mean(0)

        df = pd.DataFrame(list(zip(gesture_ids_, all_precisions, all_recalls, all_f1s)),
                          columns=['gesture_ids', 'precision', 'recall', 'f1'])
        log(df, output_folder)

        log("Overall:\tAcc\t{:.3f}\tAvg_F1\t{:.3f}\tEdit\t{:.3f}\tF1_10\t{:.3f}\tF1_25\t{:.3f}\tF1_50\t{:.3f}".format(
            np.mean(overall_acc), np.mean(overall_avg_f1), np.mean(overall_edit),
            np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50)
        ), output_folder)
        if upload:
            wandb.log({'validation accuracy': np.mean(overall_acc), 
                       'Avg_F1'             : np.mean(overall_avg_f1),
                       'Edit'               : np.mean(overall_edit), 
                       "F1_10"              : np.mean(overall_f1_10), 
                       "F1_25"              : np.mean(overall_f1_25), 
                       "F1_50"              : np.mean(overall_f1_50)},
                       step=epoch+1)

    model.train()
    return np.mean(overall_acc), np.mean(overall_avg_f1), np.mean(overall_edit), np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50), results_per_vedo

def save_fetures(model,val_loaders,list_of_videos_names,device_gpu,features_path):
    video_features = []
    all_names = []

    model.eval()
    with torch.no_grad():

        for video_num, val_loader in enumerate(val_loaders):
            video_name = val_loader.dataset.video_name
            file_path = os.path.join(features_path,video_name+".npy")

            for i, batch in enumerate(val_loader):
                data, target = batch
                data = data.to(device_gpu)
                output = model(data)
                features = output[1]
                features = features.detach().cpu().numpy()
                video_features.append(features)
            print(len(video_features))
            embedding = np.concatenate(video_features, axis=0).transpose()
            np.save(file_path,embedding)
            video_features =[]

def main(split =3,upload =False,save_features=False):
    features_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', args.dataset, 'features', f'fold {split}')
    if os.path.exists(features_path):
        print(f"Features already extracted to:\n\t'{features_path}'\nDo you want to delete them? (y/n)")
        if input() == "y":
            import shutil
            shutil.rmtree(features_path)
        else:
            print("Please delete the existing folder before running the code.")
            return
    os.makedirs(features_path, exist_ok=False)
        

    eval_metric = "F1"
    best_metric =0
    best_epoch =0
    all_eval_results =[]

    print(torch.__version__)
    print(torchvision.__version__)

    # torch.backends.cudnn.enabled = False

    torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        print("GPU not found - exit")
        return
    # args = parser.parse_args()
    args.eval_batch_size = 2 * args.batch_size
    args.split = split
    if upload:
        wandb.init(project=args.project_name, name=args.exp + "_split_" + str(split), config=args)
        wandb.config.update(args, allow_val_change=True)

    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    checkpoint = None
    if args.resume_exp:
        output_folder = args.resume_exp
    else:
        # output_folder = os.path.join(args.out, args.dataset, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
        #                               str(split), datetime.datetime.now().strftime("%H%M"))
        output_folder = os.path.join(args.out, args.dataset, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
                                      str(split))
        os.makedirs(output_folder, exist_ok=True)

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
        log("Used parameters...", output_folder)
        for arg in sorted(vars(args)):
            log("\t" + str(arg) + " : " + str(getattr(args, arg)), output_folder)

    args_dict = {}
    for arg in vars(args):
        args_dict[str(arg)] = getattr(args, arg)

    seed = 1538574472
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    if checkpoint:
        torch.set_rng_state(checkpoint['rng'])

    # ===== prepare model =====

    if args.arch == "EfficientnetV2":
        model = EfficientnetV2(size="m",num_classes=args.num_classes,pretrained=True)
    else:
        model = resnet18(pretrained=True, progress=True, num_classes=args.num_classes)

    if checkpoint:
        # load model weights
        model.load_state_dict(checkpoint['model_weights'])

    log("param count: {}".format(sum(p.numel() for p in model.parameters())), output_folder)
    log("trainable params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)), output_folder)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

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
    # TODO: add the reader for each dataset, make it global 
    if args.dataset == "VTS":
        list_of_train_examples, list_of_valid_examples, list_of_test_examples = read_VTS_data(folds_folder, split)
    elif args.dataset == "JIGSAWS":
        list_of_train_examples, list_of_valid_examples, list_of_test_examples = read_JIGSAWS_data(folds_folder, split)
    elif args.dataset == "MultiBypass140":
        list_of_train_examples, list_of_valid_examples, list_of_test_examples = read_MultiBypass140_data(folds_folder, split)
    elif args.dataset == "SAR_RARP50":
        list_of_train_examples, list_of_valid_examples, list_of_test_examples = read_SAR_RARP50_data(folds_folder, split)
    else:
        raise NotImplementedError
    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(crop_corners=args.corner_cropping,
                                                do_horizontal_flip=args.do_horizontal_flip)


    train_set = Gesture2dTrainSet(list_of_train_examples,
                                  args.data_path , 
                                  args.transcriptions_dir, 
                                  gesture_ids,
                                  image_tmpl        = args.image_tmpl,
                                  sampling_factor   = args.video_sampling_step, 
                                  video_suffix      = args.video_suffix,
                                  transform         = train_augmentation, 
                                  normalize         = normalize, 
                                  epoch_size        = (args.number_of_samples_per_class * args.num_classes), 
                                  debag             = False)


    def init_train_loader_worker(worker_id):
        np.random.seed(int((torch.initial_seed() + worker_id) % (2**32)))  # account for randomness
    
    def no_none_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size       = args.batch_size, 
                                               shuffle          = True,
                                               num_workers      = args.workers, 
                                               worker_init_fn   = init_train_loader_worker,
                                               collate_fn       = no_none_collate)
    log("Training set: will sample {} gesture snippets per pass".format(train_loader.dataset.__len__()), output_folder)


    test_loaders =[]
    val_loaders=[]

    val_augmentation = torchvision.transforms.Compose([GroupScale(int(256)),
                                                       GroupCenterCrop(args.input_size)])   ## need to be corrected

    for video in list_of_valid_examples:

        data_set = Sequential2DTestGestureDataSet(root_path             = args.data_path,
                                                  video_id              = video,
                                                  transcriptions_dir    = args.transcriptions_dir, 
                                                  gesture_ids           = gesture_ids,
                                                  snippet_length        = 1,
                                                  sampling_step         = 6,
                                                  image_tmpl            = args.image_tmpl,
                                                  video_suffix          = args.video_suffix,
                                                  normalize             = normalize,
                                                  transform             = val_augmentation)  ##augmentation are off
        val_loaders.append(torch.utils.data.DataLoader(data_set, 
                                                       batch_size       = args.eval_batch_size,
                                                       shuffle          = False, 
                                                       num_workers      = args.workers,
                                                       collate_fn       = no_none_collate))

    for video in list_of_test_examples:
        data_set = Sequential2DTestGestureDataSet(root_path             = args.data_path, 
                                                  video_id              = video,
                                                  transcriptions_dir    = args.transcriptions_dir,
                                                  gesture_ids           = gesture_ids,
                                                  snippet_length        = 1,
                                                  sampling_step         = 6,
                                                  image_tmpl            = args.image_tmpl,
                                                  video_suffix          = args.video_suffix,
                                                  normalize             = normalize,
                                                  transform             = val_augmentation)  ##augmentation are off
        test_loaders.append(torch.utils.data.DataLoader(data_set, 
                                                        batch_size      = args.eval_batch_size,
                                                        shuffle         = False, 
                                                        num_workers     = args.workers,
                                                        collate_fn      = no_none_collate))

    # ===== train model =====

    log("Start training...", output_folder)

    model = model.to(device_gpu)

    start_epoch = 0
    if checkpoint:
        start_epoch = checkpoint['epoch']
    for epoch in range(start_epoch, args.epochs):

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
                # target = target.to(dtype=torch.float)
                if args.arch == "EfficientnetV2":
                    features = output[1]
                    output = output[0]




                loss = criterion(output, target)

                loss = torch.mean(loss)

                loss.backward()
                optimizer.step()

                train_loss.update(loss.item(), batch_size)

                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)
                # _, predicted = torch.max(output, 1)

                acc = (predicted == target).sum().item() / batch_size

                train_acc.update(acc, batch_size)
                pbar.update(1)



            pbar.close()
            if scheduler is not None:
                scheduler.step()

        log("Epoch {}: Train loss: {train_loss.avg:.4f} Train acc: {train_acc.avg:.3f}"
            .format(epoch, train_loss=train_loss, train_acc=train_acc), output_folder)
        if upload:
            wandb.log({'train accuracy': train_acc.avg, 'loss': train_loss.avg}, step=epoch+1)

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            log("Start evaluation...", output_folder)

            acc, f1, f1_10, f1_25, f1_50, valid_per_video = eval(model,val_loaders,device_gpu,device_cpu,args.num_classes,output_folder,gesture_ids,epoch, upload=upload)
            all_eval_results.append([split, epoch, acc, f1, f1_10, f1_25, f1_50])
            full_eval_results = pd.DataFrame(all_eval_results,columns=['split num', 'epoch', 'acc', 'f1_macro', 'f1_10', 'f1_25', 'f1_50'])
            full_eval_results.to_csv(output_folder + "/" + "evaluation_results.csv", index=False)

            if eval_metric == "F1" and f1 > best_metric:
                best_metric = f1
                best_epoch = epoch
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
        torch.save(current_state, checkpoint_file)


    model.load_state_dict(torch.load(model_file))
    log("",output_folder)
    log("testing based on epoch " + str(best_epoch), output_folder) # based on epoch XX model
    acc_test, f1_test, f1_10_test, f1_25_test, f1_50_test, test_per_video = eval(model, test_loaders, device_gpu, device_cpu, args.num_classes, output_folder, gesture_ids,best_epoch, upload=False)
    full_test_results = pd.DataFrame(test_per_video, columns=['video name', 'acc', 'f1_macro', 'f1_10', 'f1_25', 'f1_50'])
    full_test_results["epoch"] = best_epoch
    full_test_results["split"] = split
    full_test_results.to_csv(output_folder + "/" + "test_results.csv", index=False)

    if save_features is True:
        log("Start  features saving...", output_folder)

    ### extract Features
        all_loaders =[]
        all_videos = list_of_train_examples + list_of_valid_examples + list_of_test_examples

        for video in all_videos:
            data_set = Sequential2DTestGestureDataSet(root_path=args.data_path, video_id=video,
                                                      transcriptions_dir=args.transcriptions_dir,
                                                      gesture_ids=gesture_ids,
                                                      snippet_length=1,
                                                      sampling_step=1,
                                                      image_tmpl=args.image_tmpl,
                                                      video_suffix=args.video_suffix,
                                                      normalize=normalize,
                                                      transform=val_augmentation)  ##augmentation are off
            all_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=1,
                                                           shuffle=False, num_workers=args.workers))

        save_fetures(model, all_loaders,all_videos, device_gpu,features_path)


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if args.split_num is not None:
        main(split=args.split_num,upload=args.wandb,save_features=True)
    else:
        for split in range(num_of_splits):
            main(split=split,upload=args.wandb,save_features=True)

