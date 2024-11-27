# Copyright (C) 2019  National Center of Tumor Diseases (NCT) Dresden, Division of Translational Surgical Oncology
#----------------- Python Libraries Imports -----------------#
# Python Standard Library
import datetime
import os.path
import random
import string

# Third-party libraries
import numpy as np
import pandas as pd
import PIL
import shutil
import torch
import torchvision
from torch.autograd import Variable
import tqdm
import wandb
#------------------ Bounded Future Imports ------------------#
from FE_args_2D import parser
# 16-07-2024 gabriel commented the following line #
# from utils.dataset import Gesture2dTrainSet, Sequential2DTestGestureDataSet
# 16-07-2024 gabriel added the following line #
from utils.train_dataset import Gesture2DTrainSet, Sequential2DTestGestureDataSet
from utils.efficientnetV2 import EfficientNetV2
from utils.metrics import accuracy, average_F1, edit_score, overlap_f1
from utils.resnet2D import resnet18
from utils.transforms import GroupNormalize, GroupScale, GroupCenterCrop
import utils.util
from utils.util import AverageMeter, splits_LOSO, splits_LOUO, gestures_SU
from utils.util import splits_SAR_RARP50, gestures_SAR_RARP50, splits_MultiBypass140, gestures_MultiBypass140
from utils.util import WANDB_API_KEY
#------------------------------------------------------------#
wandb.login(key=WANDB_API_KEY)

INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]
# 16-07-2024 gabriel commented the following lines
# gesture_ids = ['G0', 'G1', 'G2', 'G3', 'G4', 'G5']
# folds_folder = os.path.join("data", "APAS", "folds")


# 16-07-2024 gabriel commented this function
# def read_data(folds_folder,split_num):
#     list_of_train_examples = []
#     number_of_folds = 0
#     for file in os.listdir(folds_folder):
#         filename = os.fsdecode(file)
#         if filename.endswith(".txt") and "fold" in filename:
#             number_of_folds = number_of_folds + 1

#     for file in os.listdir(folds_folder):
#         filename = os.fsdecode(file)
#         if filename.endswith(".txt") and "fold" in filename:
#             if str(split_num) in filename:
#                 file_ptr = open(os.path.join(folds_folder, filename), 'r')
#                 list_of_test_examples = file_ptr.read().split('\n')[:-1]
#                 file_ptr.close()
#                 random.shuffle(list_of_test_examples)
#             elif str((split_num + 1) % number_of_folds) in filename:
#                 file_ptr = open(os.path.join(folds_folder, filename), 'r')
#                 list_of_examples = file_ptr.read().split('\n')[:-1]
#                 list_of_valid_examples = list_of_examples[0:12]
#                 random.shuffle(list_of_valid_examples)
#                 list_of_train_examples = list_of_train_examples + list_of_examples[12:]

#                 file_ptr.close()
#             else:
#                 file_ptr = open(os.path.join(folds_folder, filename), 'r')
#                 list_of_train_examples = list_of_train_examples + file_ptr.read().split('\n')[:-1]
#                 file_ptr.close()
#             continue
#         else:
#             continue

#     random.shuffle(list_of_train_examples)
#     return list_of_train_examples, list_of_valid_examples, list_of_test_examples


def log(msg,output_folder):
    f_log = open(os.path.join(output_folder, "log.txt"), 'a')
    utils.util.log(f_log, msg)
    f_log.close()

def eval(model, val_loaders, device_gpu, device_cpu, num_class, output_folder, gesture_ids, epoch, upload=False):
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

                Y = np.append(Y, target.numpy())
                data = data.to(device_gpu)
                output = model(data)
                if model.arch == "EfficientNetV2":
                    output = output[0]
                # TODO 16-07-2024 gabriel commented the following lines #
                # if len(output.shape) > 2:
                #     output = output[:, :, -1]  # consider only final prediction
                predicted = torch.nn.Softmax(dim=1)(output)
                _, predicted = torch.max(predicted, 1)

                P = np.append(P, predicted.to(device_cpu).numpy())

            acc = accuracy(P, Y)
            mean_avg_f1, avg_precision, avg_recall, avg_f1 = average_F1(P, Y, n_classes=num_class)
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
            log("Trial {}:\tAcc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1@10 - {:.3f} F1@25 - {:.3f} F1@50 - {:.3f}"
                .format(val_loader.dataset.video_id, acc, mean_avg_f1, edit, f1_10, f1_25, f1_50), output_folder)
            results_per_vedo.append([val_loader.dataset.video_name, acc, mean_avg_f1])

            overall_acc.append(acc)
            overall_avg_f1.append(mean_avg_f1)
            overall_edit.append(edit)
            overall_f1_10.append(f1_10)
            overall_f1_25.append(f1_25)
            overall_f1_50.append(f1_50)

        log("Overall: Validation Acc - {:.3f} F1-Macro - {:.3f} Edit - {:.3f} F1@10 - {:.3f} F1@25 - {:.3f} F1@50 - {:.3f}".format(
            np.mean(overall_acc),   np.mean(overall_avg_f1),    np.mean(overall_edit),
            np.mean(overall_f1_10), np.mean(overall_f1_25),     np.mean(overall_f1_50) ), output_folder)
        # 21-07-2024 gabriel commented the following lines #
        # gesture_ids_ = gesture_ids.copy() + ["mean"]
        # all_precisions = np.array(avg_precision_).mean(0)
        # all_recalls =  np.array(avg_recall).mean(0)
        # all_f1s = np.array(avg_f1).mean(0)

        # df = pd.DataFrame(list(zip(gesture_ids_, all_precisions, all_recalls, all_f1s)),
        #                   columns=['gesture_ids', 'precision', 'recall', 'f1'])
        # log(df, output_folder)

        log("Overall: Acc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1@10 - {:.3f} F1@25 - {:.3f} F1@50 - {:.3f}".format(
            np.mean(overall_acc), np.mean(overall_avg_f1), np.mean(overall_edit),
            np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50)
        ), output_folder)
        
        if upload:
            wandb.log({'Validation Acc': np.mean(overall_acc), 'F1-Macro': np.mean(overall_avg_f1), 
                       "Edit": np.mean(overall_edit), "F1@10": np.mean(overall_f1_10), "F1@25": np.mean(overall_f1_25),
                       "F1@50": np.mean(overall_f1_50)}, step=epoch)
        
        overall_acc_mean = np.mean(overall_acc)
        overall_avg_f1_mean = np.mean(overall_avg_f1)

    # TODO 16-07-2024 gabriel commented the following line #
    # model.train()
    return overall_acc_mean, overall_avg_f1_mean, results_per_vedo

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

def get_gestures(dataset, task=None):
    if dataset == "JIGSAWS":
        gesture_ids = gestures_SU
    elif dataset == "SAR_RARP50":
        gesture_ids = gestures_SAR_RARP50
    elif dataset == "MultiBypass140":
        gesture_ids = gestures_MultiBypass140
    else:
        raise NotImplementedError()

    return gesture_ids

def get_splits(dataset, eval_scheme, task):
    splits = None
    if dataset == "JIGSAWS":
        if eval_scheme == 'LOSO':
            splits = splits_LOSO
        elif eval_scheme == 'LOUO':
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
        train_list = splits[0:val_split] + splits[val_split + 1:]
        val_list = splits[val_split:val_split + 1]
        test_list = None

        if isinstance(train_list[0], list):
            train_list = [item for train_split in train_list for item in train_split]
        if isinstance(val_list[0], list):
            val_list = [item for val_split in val_list for item in val_split]
    elif MB140:
        assert (val_split >= 0 and val_split < len(splits['train']))
        train_list  = splits['train'][val_split]
        val_list    = splits['val'][val_split]
        test_list   = splits['test'][val_split]
    # else:
    #     assert isinstance(val_split, List)
    #     assert all((s in splits) for s in val_split)
    #     train_list = [s for s in splits if s not in val_split]
    #     val_list = [s for s in splits if s in val_split]

    return train_list, val_list, test_list

def main(split =3, upload =False, save_features=False):
    # features_path = os.path.join("data", "APAS", "features","fold "+str(split)) # 21-07-2024 gabriel commented this line: moved it to the end of the function
    group = None # 21-07-2024 gabriel added this line
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
        # 16-07-2024 gabriel commented the following lines #
        # output_folder = os.path.join(args.out, args.exp + "_" + datetime.datetime.now().strftime("%Y%m%d"),
        #                              args.eval_scheme, str(args.split), datetime.datetime.now().strftime("%H%M"))
        # os.makedirs(output_folder, exist_ok=True)
        # 16-07-2024 gabriel added the following lines #
        if args.dataset in ['JIGSAWS', 'SAR_RARP50', 'MultiBypass140']:
            output_folder = os.path.join(args.out, 'feature_extractor',
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
    gesture_ids = get_gestures(args.dataset, args.task)

    num_class = len(gesture_ids)
    if args.arch == "EfficientNetV2":
        model = EfficientNetV2(size="m",num_classes=args.num_classes, pretrained=True)
    else:
        model = resnet18(pretrained=True, progress=True, num_classes=args.num_classes)

    if checkpoint:
        # load model weights
        model.load_state_dict(checkpoint['model_weights'])

    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log("param count: {}".format(param_count), output_folder)
    log("trainable params: {}".format(trainable_params), output_folder)
    criterion = torch.nn.CrossEntropyLoss()
    # 16-07-2024 gabriel commented the following line #
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 16-07-2024 gabriel added the following line #
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
        train_list, val_list, test_list = train_val_split(splits, args.split)
        train_list = list(map(lambda x: os.path.join(lists_dir, x), train_list))
        val_list   = list(map(lambda x: os.path.join(lists_dir, x), val_list))
        if args.dataset == "SAR_RARP50":
            test_list = 'data_test.csv'
            test_list = list(map(lambda x: os.path.join(lists_dir, x), test_list.split()))
    elif args.dataset == "MultiBypass140":
        train_list, val_list, test_list = train_val_split(splits, args.split, MB140=True)
        train_list = list(map(lambda x: os.path.join(lists_dir, x), train_list.split()))
        val_list   = list(map(lambda x: os.path.join(lists_dir, x), val_list.split()))
        test_list  = list(map(lambda x: os.path.join(lists_dir, x), test_list.split()))
    # 16-07-2024 gabriel commented the following line #
    # list_of_train_examples, list_of_valid_examples, list_of_test_examples = read_data(folds_folder,split)
    # 16-07-2024 gabriel added the following lines #
    log("Splits in train set :" + str(train_list), output_folder)
    log("Splits in valid set :" + str(val_list), output_folder)  
    if test_list is not None:
        log("Splits in test set :" + str(test_list), output_folder) 
    
    normalize = GroupNormalize(model.input_mean, model.input_std)
    train_augmentation = model.get_augmentation(crop_corners        = args.corner_cropping,
                                                do_horizontal_flip  = args.do_horizontal_flip)

    # 16-07-2024 gabriel commented the following lines #
    # train_set = Gesture2dTrainSet(list_of_train_examples,args.data_path , args.transcriptions_dir, gesture_ids,
    #                            image_tmpl=args.image_tmpl,samoling_factor=args.video_sampling_step, video_suffix=args.video_suffix,
    #                            transform=train_augmentation, normalize=normalize, epoch_size = args.epoch_size, debag=False)
    # 16-07-2024 gabriel added the following lines #
    train_set = Gesture2DTrainSet(args.dataset, 
                                  args.data_path, 
                                  train_list, 
                                  args.transcriptions_dir, 
                                  gesture_ids,
                                  image_tmpl                    = args.image_tmpl, 
                                  video_suffix                  = args.video_suffix,
                                  transform                     = train_augmentation, 
                                  normalize                     = normalize, 
                                  resize                        = args.input_size, 
                                  debag                         = False,
                                  number_of_samples_per_class   = args.number_of_samples_per_class, 
                                  preload                       = args.preload)
    def no_none_collate(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)
    def init_train_loader_worker(worker_id):
        np.random.seed(int((torch.initial_seed() + worker_id) % (2**32)))  # account for randomness
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size       = args.batch_size, 
                                               shuffle          = True,
                                               num_workers      = args.workers, 
                                               worker_init_fn   = init_train_loader_worker,
                                               collate_fn       = no_none_collate)
    log("Training set: will sample {} gesture snippets per pass".format(train_loader.dataset.__len__()), output_folder)

    # 16-07-2024 gabriel commented the following lines #
    # test_loaders =[]
    # val_loaders=[]
    # val_augmentation = torchvision.transforms.Compose([GroupScale(int(256)),
    #                                                    GroupCenterCrop(args.input_size)])   ## need to be corrected
    # for video in list_of_valid_examples:

    #     data_set = Sequential2DTestGestureDataSet(root_path=args.data_path,video_id=video,transcriptions_dir= args.transcriptions_dir, gesture_ids=gesture_ids,
    #                                         snippet_length=1,
    #                                         sampling_step=6,
    #                                         image_tmpl=args.image_tmpl,
    #                                         video_suffix=args.video_suffix,
    #                                         normalize=normalize,
    #                                               transform=val_augmentation)  ##augmentation are off
    #     val_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
    #                                                    shuffle=False, num_workers=args.workers))
    # for video in list_of_test_examples:
    #     data_set = Sequential2DTestGestureDataSet(root_path=args.data_path, video_id=video,
    #                                               transcriptions_dir=args.transcriptions_dir,
    #                                               gesture_ids=gesture_ids,
    #                                               snippet_length=1,
    #                                               sampling_step=6,
    #                                               image_tmpl=args.image_tmpl,
    #                                               video_suffix=args.video_suffix,
    #                                               normalize=normalize,
    #                                               transform=val_augmentation)  ##augmentation are off
    #     test_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
    #                                                    shuffle=False, num_workers=args.workers))
    # 16-07-2024 gabriel added the following lines #
    val_augmentation = torchvision.transforms.Compose([GroupScale(args.input_size), GroupCenterCrop(args.input_size)])
    
    val_videos = list()
    for list_file in val_list:
        # format should be video_id, frame_count
        val_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
    val_loaders = list()

    test_videos = list()
    if test_list is not None:
        for list_file in test_list:
            # format should be video_id, frame_count
            test_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
        test_loaders = list()
    
    for video in val_videos:
        data_set = Sequential2DTestGestureDataSet(dataset               = args.dataset, 
                                                  root_path             = args.data_path, 
                                                  sar_rarp50_sub_dir    = 'train', 
                                                  video_id              = video[0], 
                                                  frame_count           = video[1],
                                                  transcriptions_dir    = args.transcriptions_dir, 
                                                  gesture_ids           = gesture_ids,
                                                  snippet_length        = args.snippet_length,
                                                  sampling_step         = args.val_sampling_step,
                                                  image_tmpl            = args.image_tmpl,
                                                  video_suffix          = args.video_suffix,
                                                  normalize             = normalize, 
                                                  resize                = args.input_size,
                                                  transform             = val_augmentation)  # augmentation are off
        val_loaders.append(torch.utils.data.DataLoader(data_set, 
                                                       batch_size       = args.eval_batch_size,
                                                       shuffle          = False, 
                                                       num_workers      = args.workers,
                                                       collate_fn       = no_none_collate))
    if test_list is not None:
        for video in test_videos:
            data_set = Sequential2DTestGestureDataSet(dataset           = args.dataset, 
                                                      root_path         = args.data_path, 
                                                      sar_rarp50_sub_dir='test', 
                                                      video_id          = video[0], 
                                                      frame_count       = video[1], 
                                                      transcriptions_dir= args.transcriptions_dir, 
                                                      gesture_ids       = gesture_ids,
                                                      snippet_length    = args.snippet_length,
                                                      sampling_step     = args.val_sampling_step,
                                                      image_tmpl        = args.image_tmpl,
                                                      video_suffix      = args.video_suffix,
                                                      normalize         = normalize, 
                                                      resize            = args.input_size,
                                                      transform         = val_augmentation)  # augmentation are off
            test_loaders.append(torch.utils.data.DataLoader(data_set, 
                                                            batch_size  = args.eval_batch_size,
                                                            shuffle     = False, 
                                                            num_workers = args.workers,
                                                            collate_fn  = no_none_collate))
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
    log("=========================")
    log("Start training...", output_folder)
    log("=========================")
    model = model.to(device_gpu)

    start_epoch = 0
    if checkpoint:
        start_epoch = checkpoint['epoch']

    max_acc_val = 0
    max_acc_train = 0

    for epoch in range(start_epoch, args.epochs):
        if epoch > start_epoch:
            log(f"Training set: sampling {train_loader.dataset.__len__()} gesture snippets for epoch number {epoch}", output_folder)
            train_set.randomize() # resample the dataset for each epoch
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        model.train()


        with tqdm.tqdm(desc=f'{"Epoch"} ({epoch}) {"progress"}', total=int(len(train_loader))) as pbar:
            # 16-07-2024 gabriel commented the following line #
            # for batch_i, (data, target) in enumerate(train_loader):
            # 16-07-2024 gabriel added the following lines #
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
            if args.arch == "EfficientNetV2":
                features = output[1]
                output = output[0]

            loss = criterion(output, target)

            loss = torch.mean(loss)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), batch_size, loss=True)

            predicted = torch.nn.Softmax(dim=1)(output)
            _, predicted = torch.max(predicted, 1)
            # _, predicted = torch.max(output, 1)

            acc = (predicted == target).sum().item() / batch_size

            train_acc.update(acc, batch_size)
            # 16-07-2024 gabriel commented the following line #
            # pbar.update(1)

        pbar.close()
        if scheduler is not None:
            scheduler.step()

        log("Epoch {}: Train loss: {train_loss.avg:.3f} Train acc: {train_acc.avg:.3f}"
            .format(epoch, train_loss=train_loss, train_acc=train_acc), output_folder)
        if upload:
            wandb.log({'Train Acc': train_acc.avg, 'Train loss': train_loss.avg})

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            log("=========================")
            log("Start evaluation...", output_folder)
            log("=========================")

            acc, f1, _ = eval(model, val_loaders, device_gpu, device_cpu, args.num_classes, output_folder, gesture_ids, epoch, upload)
            all_eval_results.append([split, epoch, acc, f1])
            full_eval_results = pd.DataFrame(all_eval_results,columns=['split num', 'epoch', 'acc', 'f1'])
            full_eval_results.to_csv(output_folder + "/" + "evaluation_results.csv", index=False)

            if eval_metric not in ["F1", "Acc"]:
                raise NotImplementedError()
            elif eval_metric == "F1" and f1 > best_metric:
                best_metric = f1
                best_epoch = epoch
                # ===== save model =====
                model_file = os.path.join(output_folder, "model_" + str(epoch) + ".pth")
                torch.save(model.state_dict(), model_file)
                log("Saved model to " + model_file, output_folder)
            elif eval_metric == "Acc" and acc > best_metric:
                best_metric = acc
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
            if upload:
                current_state["run_id"] = wandb.run.id

    # ===== model results =====
    model.load_state_dict(torch.load(model_file))
    log("",output_folder)
    log("=========================")
    log("testing based on epoch " + str(best_epoch), output_folder) # based on epoch XX model
    log("=========================")
    # In JIGSAWS the dataset is split in train and test only
    if args.dataset == "JIGSAWS":
        _, _, test_per_video = eval(model, val_loaders, device_gpu, device_cpu, args.num_classes, output_folder, gesture_ids, best_epoch)
    else: # In SAR_RARP50 and MultiBypass140 the dataset is split in train, val and test    
        _, _, test_per_video = eval(model, test_loaders, device_gpu, device_cpu, args.num_classes, output_folder, gesture_ids, best_epoch)
    full_test_results = pd.DataFrame(test_per_video, columns=['video name', 'acc', 'f1'])
    full_test_results["epoch"] = best_epoch
    full_test_results["split"] = split
    full_test_results.to_csv(output_folder + "/" + "test_results.csv", index=False)

    if save_features is True:
        log("=========================")
        log("Start  features saving...", output_folder)
        log("=========================")

    ### extract Features
        all_loaders =[]
        # 16-07-2024 gabriel commented the following line #
        # all_videos = list_of_train_examples + list_of_valid_examples + list_of_test_examples
        if args.dataset == "JIGSAWS":
            all_videos_list = train_list + val_list
        else:
            all_videos_list = train_list + val_list + test_list

        all_videos = list()
        for list_file in all_videos_list:
            # format should be video_id, frame_count
            all_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
        all_loaders = list()

        if args.dataset in ['JIGSAWS', 'MultiBypass140']:
            sampling_step = 1
        elif args.dataset == 'SAR_RARP50':
            sampling_step = 6

        for video in all_videos:
            if 'test' in video:
                sar_rarp50_sub_dir = 'test'
            else:
                sar_rarp50_sub_dir = 'train'
            data_set = Sequential2DTestGestureDataSet(dataset               = args.dataset,
                                                      root_path             = args.data_path,
                                                      sar_rarp50_sub_dir    = sar_rarp50_sub_dir, 
                                                      video_id              = video[0],         
                                                      frame_count           = video[1],
                                                      transcriptions_dir    = args.transcriptions_dir,
                                                      gesture_ids           = gesture_ids,
                                                      snippet_length        = 1,
                                                      sampling_step         = sampling_step,
                                                      image_tmpl            = args.image_tmpl,
                                                      video_suffix          = args.video_suffix,
                                                      normalize             = normalize,
                                                      transform             = val_augmentation)  ##augmentation are off
            all_loaders.append(torch.utils.data.DataLoader(data_set, 
                                                           batch_size       = 1,
                                                           shuffle          = False, 
                                                           num_workers      = args.workers))
        features_path = os.path.join(args.out, 'features',
                                     args.dataset,
                                     args.arch,
                                     args.eval_scheme, 
                                     str(args.split))
        save_fetures(model, all_loaders, all_videos, device_gpu, features_path)


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
    # Run Main
    for i in range(user_num):
        main(split=i, upload=args.upload, save_features=True)

def run_full_LOSO(group_name=None):
    args = parser.parse_args()

    if args.dataset in ['VTS', 'MultiBypass140', 'SAR_RARP50']:
        raise NotImplementedError(f"{args.dataset} not implemented")
    elif args.dataset == "JIGSAWS":
        supertrial_num = len(splits_LOSO)
    # Run Main
    for i in range(supertrial_num):
        main(split=i, upload=args.upload, save_features=True)

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # main(split=0,save_features=False)
    # main(split=1,save_features=False)
    # main(split=2,save_features=False)
    # main(split=3,save_features=False)
    # main(split=4,save_features=False)

    args = parser.parse_args()
    if args.dataset in ['VTS']:
        raise NotImplementedError(f"{args.dataset} not implemented")

    elif args.dataset in ['JIGSAWS', 'SAR_RARP50', 'MultiBypass140']:
        if args.split_num is not None:
            main(split=args.split_num, upload=args.upload, save_features=True)
        elif args.eval_scheme == "LOUO" or args.dataset in ['SAR_RARP50', 'MultiBypass140']:
            run_full_LOUO()
        elif args.eval_scheme == "LOSO":
            run_full_LOSO()
