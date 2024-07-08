#----------------- Python Libraries Imports -----------------#
# Python Standard Library

# Third-party libraries
import argparse
#------------------ Bounded Future Imports ------------------#
from FeatureExtractorTrainer import *
#------------------------------------------------------------#

# args for testing the model
parser = argparse.ArgumentParser()

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--arch', type=str, default='2D-EfficientNetV2-m')
parser.add_argument('--video_lists_dir', type=str, default="/data/home/gabrielg/Bounded_Future_from_GIT/data/SAR_RARP50/Splits")
parser.add_argument('--data_path', type=str, default="/data/home/gabrielg/Bounded_Future_from_GIT/data/SAR_RARP50/frames")
parser.add_argument('--transcriptions_dir', type=str, default="/data/home/gabrielg/Bounded_Future_from_GIT/data/SAR_RARP50/transcriptions")
parser.add_argument('--model_path', type=str, default="/data/home/gabrielg/Bounded_Future_from_GIT/output/feature_extractor")
parser.add_argument('--dataset', type=str, default='SAR_RARP50')  # 'JIGSAWS' or MultiBypass140
parser.add_argument('--num_classes', type=int, default=8)  # 10 for JIGSAWS, 8 for SAR_RARP50, n for MultiBypass
parser.add_argument('--eval_scheme', type=str, default='LOUO')  # LOUO or LOSO
parser.add_argument('--task', type=str, default='None')  # 'Suturing' for JIGSAWS and 'None' for SAR_RARP50
parser.add_argument('--split', type=int, default=0)
parser.add_argument('--snippet_length', type=int, default=1)
parser.add_argument('--val_sampling_step', type=int, default=6) # multiply of 6 for SAR_RARP50 (60 fps video, 10 Hz labels => each 6 frames there's a label)
parser.add_argument('--image_tmpl', type=str, choices=['img_{:05d}.jpg', '{:09d}.png'], default='{:09d}.png')
parser.add_argument('--video_suffix', type=str, choices=['_capture1', '_capture2', 'None'], default='None') # _capture* for JIGSAWS, '' for SAR_RARP50
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--workers', type=int, default=64)

args = parser.parse_args()

class Overall:
    def __init__(self):
        self.acc_mean = None
        self.edit_mean = None
        self.avg_f1_mean = None
        self.f1_10_mean = None
        self.f1_25_mean = None
        self.f1_50_mean = None

def test(model, test_loaders, device_gpu, device_cpu, num_class, gesture_ids, output_folder=None, epoch=None, upload=False):
    model.eval()
    with torch.no_grad():

        overall_acc = []
        overall_avg_f1 = []
        overall_edit = []
        overall_f1_10 = []
        overall_f1_25 = []
        overall_f1_50 = []

        overall = Overall()  # Initialize overall as an object of Overall class

        for val_loader in test_loaders:
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
            if output_folder:
                log(df, output_folder)
            edit = edit_score(P, Y)
            f1_10 = overlap_f1(P, Y, n_classes=num_class, overlap=0.1)
            f1_25 = overlap_f1(P, Y, n_classes=num_class, overlap=0.25)
            f1_50 = overlap_f1(P, Y, n_classes=num_class, overlap=0.5)
            if output_folder:
                log("Trial {}:\tAcc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1_10 {:.3f} F1_25 {:.3f} F1_50 {:.3f}"
                    .format(val_loader.dataset.video_id, acc, mean_avg_f1, edit, f1_10, f1_25, f1_50), output_folder)

            overall_acc.append(acc)
            overall_avg_f1.append(mean_avg_f1)
            overall_edit.append(edit)
            overall_f1_10.append(f1_10)
            overall_f1_25.append(f1_25)
            overall_f1_50.append(f1_50)
        if output_folder:
            log("Overall: Acc - {:.3f} Avg_F1 - {:.3f} Edit - {:.3f} F1_10 {:.3f} F1_25 {:.3f} F1_50 {:.3f}".format(
                np.mean(overall_acc), np.mean(overall_avg_f1), np.mean(overall_edit),
                np.mean(overall_f1_10), np.mean(overall_f1_25), np.mean(overall_f1_50)
            ), output_folder)

        
        if upload:
            wandb.log({'validation accuracy': np.mean(overall_acc), 'Avg_F1': np.mean(overall_avg_f1), 
                        'Edit': np.mean(overall_edit), "F1_10": np.mean(overall_f1_10), "F1_25": np.mean(overall_f1_25),
                        "F1_50": np.mean(overall_f1_50)}, step=epoch)
        overall.acc_mean    = np.mean(overall_acc)
        overall.edit_mean   = np.mean(overall_edit)
        overall.avg_f1_mean = np.mean(overall_avg_f1)
        overall.f1_10_mean  = np.mean(overall_f1_10)
        overall.f1_25_mean  = np.mean(overall_f1_25)
        overall.f1_50_mean  = np.mean(overall_f1_50)

    return overall

def no_none_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    # get total splits
    if args.eval_scheme == 'LOUO':
        if args.dataset == 'JIGSAWS':
            total_splits = 8
        elif args.dataset == 'SAR_RARP50':
            total_splits = 5
    elif args.eval_scheme == 'LOSO':
        total_splits = 5 # only for JIGSAWS

    # for results.csv
    results = pd.DataFrame(columns=["split", "test_acc", "test_edit", "test_macro_f1", "test_f1_10", "test_f1_25", "test_f1_50"])

    # Run for all splits
    for i in range(total_splits):
        print(f"=========================")
        print(f"Testing Split Num: {i}...")
        print(f"=========================")
        # ===== load data =====
        gesture_ids = get_gestures(args.dataset, args.task)
        args.eval_batch_size = 2 * args.batch_size
        normalize = GroupNormalize(INPUT_MEAN, INPUT_STD)

        if args.dataset == "JIGSAWS":
            splits = get_splits(args.dataset, args.eval_scheme, args.task)
            _, test_list = train_val_split(splits, args.split)
            lists_dir = os.path.join(args.video_lists_dir, args.eval_scheme)
        elif args.dataset == "SAR_RARP50":
            test_list = {'data_test.csv'}
            lists_dir = args.video_lists_dir
        else:
            raise NotImplementedError()

        val_augmentation = torchvision.transforms.Compose([GroupScale(args.input_size), GroupCenterCrop(args.input_size)])
        test_lists = list(map(lambda x: os.path.join(lists_dir, x), test_list))

        test_videos = list()
        for list_file in test_lists:
            test_videos.extend([(x.strip().split(',')[0], x.strip().split(',')[1]) for x in open(list_file)])
        test_loaders = list()
        # in JIGSAWS there is no validation, so each split the test set changes
        if (args.dataset == "JIGSAWS") or (args.dataset == "SAR_RARP50" and i==0):
            for video in test_videos:
                data_set = Sequential2DTestGestureDataSet(dataset=args.dataset, root_path=args.data_path, sar_rarp50_sub_dir='test', video_id=video[0], frame_count=video[1],
                                                            transcriptions_dir=args.transcriptions_dir, gesture_ids=gesture_ids,
                                                            snippet_length=args.snippet_length,
                                                            sampling_step=args.val_sampling_step,
                                                            image_tmpl=args.image_tmpl,
                                                            video_suffix=args.video_suffix,
                                                            normalize=normalize, resize=args.input_size,
                                                            transform=val_augmentation)  # augmentation are off
                test_loaders.append(torch.utils.data.DataLoader(data_set, batch_size=args.eval_batch_size,
                                                                shuffle=False, num_workers=args.workers,
                                                                collate_fn=no_none_collate))
        
        model = get_model(  args.arch, 
                            num_classes=args.num_classes,
                            add_layer_param_num=0,
                            add_certainty_pred=0,
                            input_shape=0,
                            embedding_shape=0,
                            vae_intermediate_size=None
                        )      


        # load best model weights from output folder
        # best_model_loc = f"/data/home/gabrielg/Bounded_Future_from_GIT/output/feature_extractor/{args.dataset}/{args.arch}/{args.eval_scheme}/{args.split}/best_{args.split}.pth"
        model_loc = f"{args.model_path}/{args.dataset}/{args.arch}/{args.eval_scheme}/{args.split}/model_99.pth"
        model.load_state_dict(torch.load(model_loc))

        # model
        device_gpu = torch.device(f"cuda:{args.gpu_id}")
        model = model.to(device_gpu)
        device_cpu = torch.device("cpu")

        overall = test(model, test_loaders, device_gpu, device_cpu, args.num_classes, gesture_ids, output_folder=None, epoch=None, upload=False)

        test_acc        = overall.acc_mean
        test_edit       = overall.edit_mean
        test_macro_f1   = overall.avg_f1_mean
        test_f1_10      = overall.f1_10_mean
        test_f1_25      = overall.f1_25_mean
        test_f1_50      = overall.f1_50_mean

        # print in blue text and in yellow results including args.split
        print("\033[94m" + "Split " + "\033[93m" + f"{args.split}" + "\033[94m" + ":" + "\033[0m")
        print("\033[94m" + "\tTest Acc: " + "\033[93m" + f"\t{test_acc:.3f}" + "\033[0m")
        print("\033[94m" + "\tTest Edit: " + "\033[93m" + f"\t{test_edit:.3f}" + "\033[0m")
        print("\033[94m" + "\tTest Macro F1: " + "\033[93m" + f"\t{test_macro_f1:.3f}" + "\033[0m")
        print("\033[94m" + "\tTest F1@10: " + "\033[93m" + f"\t{test_f1_10:.3f}" + "\033[0m")
        print("\033[94m" + "\tTest F1@25: " + "\033[93m" + f"\t{test_f1_25:.3f}" + "\033[0m")
        print("\033[94m" + "\tTest F1@50: " + "\033[93m" + f"\t{test_f1_50:.3f}" + "\033[0m")

        results.loc[i] = [args.split, test_acc, test_edit, test_macro_f1, test_f1_10, test_f1_25, test_f1_50]

        args.split += 1

    # keep results in csv file
    results.to_csv(f"{args.model_path}/{args.dataset}/{args.arch}/{args.eval_scheme}/test_results.csv", index=False)