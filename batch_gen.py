#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import pandas as pd
from scipy.stats import norm


class BatchGenerator(object):
    def __init__(self, dataset, num_classes_gestures, num_classes_tools, actions_dict_gestures, actions_dict_tools, features_path, split_num, folds_folder, gt_path_gestures=None, gt_path_tools_left=None, gt_path_tools_right=None, sample_rate=1, normalization="None", task="gestures"):
        """
        :param dataset: VTS or JIGSAWS or MultiBypass140 or RARP150
        :param num_classes_gestures: 
        :param num_classes_tools: 
        :param actions_dict_gestures: 
        :param actions_dict_tools: 
        :param features_path: 
        :param split_num: 
        :param folds_folder: 
        :param gt_path_gestures: 
        :param gt_path_tools_left: 
        :param gt_path_tools_right: 
        :param sample_rate: 
        :param normalization: None - no normalization, min-max - Min-max feature scaling, Standard - Standard score	 or Z-score Normalization
        ## https://en.wikipedia.org/wiki/Normalization_(statistics)
        """""
        self.dataset                = dataset
        self.task                   = task
        self.normalization          = normalization
        self.folds_folder           = folds_folder
        self.split_num              = split_num
        self.list_of_train_examples = list()
        self.list_of_valid_examples = list()
        self.list_of_test_examples  = list()
        self.index                  = 0
        self.num_classes_gestures   = num_classes_gestures
        self.num_classes_tools      = num_classes_tools
        self.actions_dict_gestures  = actions_dict_gestures
        self.action_dict_tools      = actions_dict_tools
        self.gt_path_gestures       = gt_path_gestures
        self.gt_path_tools_left     = gt_path_tools_left
        self.gt_path_tools_right    = gt_path_tools_right
        self.features_path          = features_path
        self.sample_rate            = sample_rate
        if self.dataset == "VTS":
            self.read_VTS_data()
            # foldfile_end_with = ".txt"
            # foldfile_in_name = "fold"
        elif self.dataset == "JIGSAWS":
            self.read_JIGSAWS_data()
            # foldfile_end_with = ".csv"
            # foldfile_in_name = "data_"
        elif self.dataset == "SAR_RARP50":
            self.read_SAR_RARP50_data()
        else: # MultiBypass140
            self.read_MultiBypass140_data()

        self.normalization_params_read()

    #---------------------------------VTS---------------------------------#
    def read_VTS_data(self):
        self.list_of_train_examples = []
        number_of_folds = 0
        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "fold" in filename:
                number_of_folds = number_of_folds + 1

        for file in os.listdir(self.folds_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "fold" in filename:
                if str(self.split_num) in filename:
                    file_ptr = open(os.path.join(
                        self.folds_folder, filename), 'r')
                    self.list_of_test_examples = file_ptr.read().split('\n')[
                        :-1]
                    file_ptr.close()
                    random.shuffle(self.list_of_test_examples)
                elif str((self.split_num + 1) % number_of_folds) in filename:
                    file_ptr = open(os.path.join(
                        self.folds_folder, filename), 'r')
                    list_of_examples = file_ptr.read().split('\n')[:-1]
                    self.list_of_valid_examples = list_of_examples[0:12]
                    random.shuffle(self.list_of_valid_examples)
                    self.list_of_train_examples = self.list_of_train_examples + \
                        list_of_examples[12:]

                    file_ptr.close()
                else:
                    file_ptr = open(os.path.join(
                        self.folds_folder, filename), 'r')
                    self.list_of_train_examples = self.list_of_train_examples + \
                        file_ptr.read().split('\n')[:-1]
                    file_ptr.close()
                continue
            else:
                continue

        random.shuffle(self.list_of_train_examples)
    #-------------------------------JIGSAWS-------------------------------#
    def read_JIGSAWS_data(self):
        self.list_of_train_examples = []
        number_of_folds = 0
        for file in sorted(os.listdir(self.folds_folder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "valid" in filename:
                number_of_folds = number_of_folds + 1

        for file in sorted(os.listdir(self.folds_folder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "valid" in filename:
                file_path = os.path.join(self.folds_folder, filename)
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
                    if str(self.split_num) in filename:
                        self.list_of_valid_examples = files_to_sort
                        if not self.list_of_valid_examples:
                            print(f"The file {file_path} only contains empty lines or ends with a newline.")
                            raise EOFError
                        else:
                            random.shuffle(self.list_of_valid_examples)
                    #----- Test set -----# Using the same data for test and validation since there is no validation set in JIGSAWS dataset
                    if str(self.split_num) in filename:
                            self.list_of_test_examples = files_to_sort
                            if not self.list_of_test_examples:
                                print(f"The file {file_path} only contains empty lines or ends with a newline.")
                                raise EOFError
                            else:
                                random.shuffle(self.list_of_test_examples)
                    #----- Training set -----#
                    else:
                        self.list_of_train_examples = self.list_of_train_examples + files_to_sort
                        if not self.list_of_train_examples:
                            print(f"The file {file_path} only contains empty lines or ends with a newline.")
                            raise EOFError
                continue
            else:
                continue

        random.shuffle(self.list_of_train_examples)
    #----------------------------MultiBypass140---------------------------#
    def read_MultiBypass140_data(self):
        self.list_of_train_examples = []
        number_of_folds = 0
        for file in sorted(os.listdir(self.folds_folder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "val" in filename:
                number_of_folds = number_of_folds + 1

        for file in sorted(os.listdir(self.folds_folder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and str(self.split_num) in filename:
                file_path = os.path.join(self.folds_folder, filename)
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
                    self.list_of_valid_examples = files_to_sort
                    if not self.list_of_valid_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
                    else:
                        random.shuffle(self.list_of_valid_examples)
                elif "train" in filename:
                    #----- Training set -----#
                    self.list_of_train_examples = files_to_sort
                    if not self.list_of_train_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
                    else:
                        random.shuffle(self.list_of_train_examples)
                elif "test" in filename:
                    #----- Test set -----# 
                    self.list_of_test_examples = files_to_sort
                    if not self.list_of_test_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
                    else:
                        random.shuffle(self.list_of_test_examples)
    #-----------------------------SAR-RARP50------------------------------#
    def read_SAR_RARP50_data(self):
        self.list_of_train_examples = []
        number_of_folds = 0
        for file in sorted(os.listdir(self.folds_folder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and "valid" in filename:
                number_of_folds = number_of_folds + 1

        for file in sorted(os.listdir(self.folds_folder)):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folds_folder, filename)
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
                    if str(self.split_num) in filename:
                        self.list_of_valid_examples = files_to_sort
                        if not self.list_of_valid_examples:
                            print(f"The file {file_path} only contains empty lines or ends with a newline.")
                            raise EOFError
                        else:
                            random.shuffle(self.list_of_valid_examples)
                    #----- Training set -----#
                    else:
                        self.list_of_train_examples = self.list_of_train_examples + files_to_sort
                        if not self.list_of_train_examples:
                            print(f"The file {file_path} only contains empty lines or ends with a newline.")
                            raise EOFError
                elif "test" in filename:
                    #----- Test set -----# 
                    self.list_of_test_examples = files_to_sort
                    if not self.list_of_test_examples:
                        print(f"The file {file_path} only contains empty lines or ends with a newline.")
                        raise EOFError
                    else:
                        random.shuffle(self.list_of_test_examples)
            else:
                continue

        random.shuffle(self.list_of_train_examples)
    #---------------------------------------------------------------------#        
    def normalization_params_read(self):
        if self.normalization == "None":
            return
        params = pd.read_csv(os.path.join(
            self.folds_folder, "std_params_fold_" + str(self.split_num) + ".csv"), index_col=0).values
        self.max = params[0, :]
        self.min = params[1, :]
        self.mean = params[2, :]
        self.std = params[3, :]

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_train_examples)

    def has_next(self):
        if self.index < len(self.list_of_train_examples):
            return True
        return False
    

    def pars_ground_truth(self, gt_source):
        contant = []
        for line in gt_source:
            if self.dataset in ["VTS", "JIGSAWS", "MultiBypass140"]:
                info = line.split()
                line_contant = [info[2]] * (int(info[1])-int(info[0]) + 1) 
            elif self.dataset in ["SAR_RARP50"]:
                info = line.split(',')
                line_contant = ['G'+info[1]]
            contant = contant + line_contant # TODO Maybe here I can export ref Labels
        return contant


##### this is supports one and two heads and 3 heads #############


    def next_batch(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input             = []
        batch_target_gestures   = []
        batch_target_left       = []
        batch_target_right      = []

        for seq in batch:
            # print(seq)
            features = np.load(os.path.join(self.features_path, seq.split('.')[0] + '.npy'))

            if self.normalization == "Min-max":
                numerator = features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator = features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T
            elif self.normalization == "samplewise_SD":
                samplewise_meam = features.mean(axis=1)
                samplewise_std = features.std(axis=1)
                numerator = features.T - samplewise_meam
                denominator = samplewise_std
                features = (numerator / denominator).T

            batch_input.append(features[:, ::self.sample_rate])

            if self.task in ["gestures", "steps", "phases"]: # TODO: 23-09-2024: I need to check this part
                if self.task == "gestures":
                    if self.dataset == "SAR_RARP50":
                        file_ptr = open(os.path.join(self.gt_path_gestures, seq.split('.')[0] + '_discrete.txt'), 'r')
                    else:
                        file_ptr = open(os.path.join(self.gt_path_gestures, seq.split('.')[0] + '.txt'), 'r')
                else:
                    file_ptr = open(os.path.join(self.gt_path_gestures, self.task, seq.split('.')[0] + '.txt'), 'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content = self.pars_ground_truth(gt_source)
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_target_gestures.append(classes[::self.sample_rate])

            elif self.task == "tools":
                raise NotImplementedError
                file_ptr_right = open(os.path.join(
                    self.gt_path_tools_right, seq.split('.')[0] + '.txt'), 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                file_ptr_left = open(os.path.join(
                    self.gt_path_tools_left, seq.split('.')[0] + '.txt'), 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)

                classes_size_right = min(np.shape(features)[1], len(
                    content_left), len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(classes_size_right):
                    classes_right[i] = self.action_dict_tools[content_right[i]]
                batch_target_right.append(classes_right[::self.sample_rate])

                classes_size_left = min(np.shape(features)[1], len(
                    content_left), len(content_right))
                classes_left = np.zeros(classes_size_left)
                for i in range(classes_size_left):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])

            elif self.task == "multi-taks":
                raise NotImplementedError
                file_ptr = open(os.path.join(
                    self.gt_path_gestures, seq.split('.')[0] + '.txt'), 'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content = self.pars_ground_truth(gt_source)
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_target_gestures.append(classes[::self.sample_rate])

                file_ptr_right = open(os.path.join(
                    self.gt_path_tools_right, seq.split('.')[0] + '.txt'), 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                classes_size_right = min(
                    np.shape(features)[1], len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(len(classes_right)):
                    classes_right[i] = self.action_dict_tools[content_right[i]]

                batch_target_right.append(classes_right[::self.sample_rate])

                file_ptr_left = open(os.path.join(
                    self.gt_path_tools_left, seq.split('.')[0] + '.txt'), 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)
                classes_size_left = min(
                    np.shape(features)[1], len(content_left))
                classes_left = np.zeros(classes_size_left)
                for i in range(len(classes_left)):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])

        if self.task in ["gestures", "steps", "phases"]: # TODO: 23-09-2024: I need to check this part
            length_of_sequences = list(map(len, batch_target_gestures))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
            batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
            mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(length_of_sequences), dtype=torch.float)
            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i][:, :batch_input_tensor.shape[2]])
                batch_target_tensor[i, :np.shape(batch_target_gestures[i])[0]] = torch.from_numpy(batch_target_gestures[i])
                mask[i, :, :np.shape(batch_target_gestures[i])[0]] = torch.ones(self.num_classes_gestures, np.shape(batch_target_gestures[i])[0])

            return batch_input_tensor, batch_target_tensor, mask

        elif self.task == "tools":
            length_of_sequences_left = np.expand_dims(
                np.array(list(map(len, batch_target_left))), 1)
            length_of_sequences_right = np.expand_dims(
                np.array(list(map(len, batch_target_right))), 1)

            length_of_sequences = list(np.min(np.concatenate(
                (length_of_sequences_left, length_of_sequences_right), 1), 1))

            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)
            batch_target_tensor_left = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long) * (-100)
            mask = torch.zeros(len(batch_target_right), self.num_classes_tools, max(
                length_of_sequences), dtype=torch.float)

            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(
                    batch_input[i][:, :batch_input_tensor.shape[2]])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(
                    batch_target_left[i][:batch_target_tensor_left.shape[1]])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(
                    batch_target_right[i][:batch_target_tensor_right.shape[1]])
                mask[i, :, :np.shape(batch_target_right[i])[0]] = torch.ones(
                    self.num_classes_tools, np.shape(batch_target_right[i])[0])

            return batch_input_tensor, batch_target_tensor_left, batch_target_tensor_right, mask

        elif self.task == "multi-taks":
            length_of_sequences_left = np.expand_dims(
                np.array(list(map(len, batch_target_left))), 1)
            length_of_sequences_right = np.expand_dims(
                np.array(list(map(len, batch_target_right))), 1)
            length_of_sequences_gestures = np.expand_dims(
                np.array(list(map(len, batch_target_gestures))), 1)

            length_of_sequences = list(np.min(np.concatenate(
                (length_of_sequences_left, length_of_sequences_right, length_of_sequences_gestures), 1), 1))

            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)

            batch_target_tensor_left = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_gestures = torch.ones(
                len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)

            mask_gesture = torch.zeros(len(batch_input), self.num_classes_gestures, max(
                length_of_sequences), dtype=torch.float)
            mask_tools = torch.zeros(len(batch_input), self.num_classes_tools, max(
                length_of_sequences), dtype=torch.float)

            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(
                    batch_input[i][:, :batch_input_tensor.shape[2]])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[0]] = torch.from_numpy(
                    batch_target_left[i][:batch_target_tensor_left.shape[1]])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[0]] = torch.from_numpy(
                    batch_target_right[i][:batch_target_tensor_right.shape[1]])
                batch_target_tensor_gestures[i, :np.shape(batch_target_gestures[i])[0]] = torch.from_numpy(
                    batch_target_gestures[i][:batch_target_tensor_gestures.shape[1]])
                mask_gesture[i, :, :length_of_sequences[i]] = torch.ones(
                    self.num_classes_gestures, length_of_sequences[i])
                mask_tools[i, :, :length_of_sequences[i]] = torch.ones(
                    self.num_classes_tools, length_of_sequences[i])

            return batch_input_tensor, batch_target_tensor_left, batch_target_tensor_right, batch_target_tensor_gestures, mask_gesture, mask_tools
    ##### this is supports one and two heads#############

    def next_batch_backup(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(self.features_path + seq.split('.')[0] + '.npy')
            if self.normalization == "Min-max":
                numerator = features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator = features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T

            if self.task == "gestures":
                file_ptr = open(self.gt_path_gestures +
                                seq.split('.')[0] + '.txt', 'r')
                gt_source = file_ptr.read().split('\n')[:-1]
                content = self.pars_ground_truth(gt_source)
                classes_size = min(np.shape(features)[1], len(content))

                classes = np.zeros(classes_size)
                for i in range(len(classes)):
                    classes[i] = self.actions_dict_gestures[content[i]]
                batch_input .append(features[:, ::self.sample_rate])
                batch_target.append(classes[::self.sample_rate])

            elif self.task == "tools":
                raise NotImplementedError
                file_ptr_right = open(
                    self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
                gt_source_right = file_ptr_right.read().split('\n')[:-1]
                content_right = self.pars_ground_truth(gt_source_right)
                classes_size_right = min(
                    np.shape(features)[1], len(content_right))
                classes_right = np.zeros(classes_size_right)
                for i in range(len(classes_right)):
                    classes_right[i] = self.action_dict_tools[content_right[i]]

                batch_input.append(features[:, ::self.sample_rate])
                batch_target_right.append(classes_right[::self.sample_rate])

                file_ptr_left = open(
                    self.gt_path_tools_left + seq.split('.')[0] + '.txt', 'r')
                gt_source_left = file_ptr_left.read().split('\n')[:-1]
                content_left = self.pars_ground_truth(gt_source_left)
                classes_size_left = min(
                    np.shape(features)[1], len(content_left))
                classes_left = np.zeros(classes_size_left)
                for i in range(len(classes_left)):
                    classes_left[i] = self.action_dict_tools[content_left[i]]

                batch_target_left.append(classes_left[::self.sample_rate])

        if self.task == "gestures":
            length_of_sequences = list(map(len, batch_target))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[
                                             0], max(length_of_sequences), dtype=torch.float)
            batch_target_tensor = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long)*(-100)
            mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(
                length_of_sequences), dtype=torch.float)
            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(
                    batch_input[i][:, :batch_input_tensor.shape[2]])
                batch_target_tensor[i, :np.shape(batch_target[i])[
                    0]] = torch.from_numpy(batch_target[i])
                mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(
                    self.num_classes_gestures, np.shape(batch_target[i])[0])

            return batch_input_tensor, batch_target_tensor, mask

        elif self.task == "tools":
            length_of_sequences = list(map(len, batch_target_left))
            batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0],
                                             max(length_of_sequences), dtype=torch.float)
            batch_target_tensor_left = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long) * (-100)
            batch_target_tensor_right = torch.ones(len(batch_input), max(
                length_of_sequences), dtype=torch.long) * (-100)
            mask = torch.zeros(len(batch_input), self.num_classes_tools, max(
                length_of_sequences), dtype=torch.float)

            for i in range(len(batch_input)):
                batch_input_tensor[i, :, :np.shape(batch_input[i])[
                    1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor_left[i, :np.shape(batch_target_left[i])[
                    0]] = torch.from_numpy(batch_target_left[i])
                batch_target_tensor_right[i, :np.shape(batch_target_right[i])[
                    0]] = torch.from_numpy(batch_target_right[i])
                mask[i, :, :np.shape(batch_target_right[i])[0]] = torch.ones(
                    self.num_classes_gestures, np.shape(batch_target_right[i])[0])

            return batch_input_tensor, batch_target_tensor_left, batch_target_tensor_right

    def next_batch_with_gt_tools_as_input(self, batch_size):
        batch = self.list_of_train_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_target_left = []
        batch_target_right = []

        for seq in batch:
            features = np.load(self.features_path + seq.split('.')[0] + '.npy')
            if self.normalization == "Min-max":
                numerator = features.T - self.min
                denominator = self.max-self.min
                features = (numerator / denominator).T
            elif self.normalization == "Standard":
                numerator = features.T - self.mean
                denominator = self.std
                features = (numerator / denominator).T

            file_ptr = open(self.gt_path_gestures +
                            seq.split('.')[0] + '.txt', 'r')
            gt_source = file_ptr.read().split('\n')[:-1]
            content = self.pars_ground_truth(gt_source)
            classes_size = min(np.shape(features)[1], len(content))

            classes = np.zeros(classes_size)
            for i in range(len(classes)):
                classes[i] = self.actions_dict_gestures[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

            file_ptr_right = open(
                self.gt_path_tools_right + seq.split('.')[0] + '.txt', 'r')
            gt_source_right = file_ptr_right.read().split('\n')[:-1]
            content_right = self.pars_ground_truth(gt_source_right)
            classes_size_right = min(np.shape(features)[1], len(content_right))
            classes_right = np.zeros(classes_size_right)
            for i in range(len(classes_right)):
                classes_right[i] = self.action_dict_tools[content_right[i]]

            batch_target_right.append(classes_right[::self.sample_rate])

            file_ptr_left = open(self.gt_path_tools_left +
                                 seq.split('.')[0] + '.txt', 'r')
            gt_source_left = file_ptr_left.read().split('\n')[:-1]
            content_left = self.pars_ground_truth(gt_source_left)
            classes_size_left = min(np.shape(features)[1], len(content_left))
            classes_left = np.zeros(classes_size_left)
            for i in range(len(classes_left)):
                classes_left[i] = self.action_dict_tools[content_left[i]]

            batch_target_left.append(classes_left[::self.sample_rate])

        # for i in range(len(batch_input)):
        #     min_dim = min([batch_target_left[i].size,batch_target_right[i].size, batch_input[i].shape[1]])
        #     batch_target_left[i] = (np.expand_dims(batch_target_left[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
        #     batch_target_right[i] = (np.expand_dims(batch_target_right[i][:min_dim], axis=1).T)/ max(self.action_dict_tools.values())
        #     batch_input[i] = np.concatenate((batch_input[i][:,:min_dim],batch_target_right[i],batch_target_left[i]), axis=0 )

        for i in range(len(batch_input)):
            min_dim = min([batch_target_left[i].size,
                          batch_target_right[i].size, batch_input[i].shape[1]])
            batch_target_left[i] = (np.expand_dims(
                batch_target_left[i][:min_dim], axis=1).T) / max(self.action_dict_tools.values())
            batch_target_right[i] = (np.expand_dims(
                batch_target_right[i][:min_dim], axis=1).T) / max(self.action_dict_tools.values())
            batch_input[i] = np.concatenate(
                (batch_target_right[i], batch_target_left[i]), axis=0)

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[
                                         0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(
            length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes_gestures, max(
            length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[
                1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[
                0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(
                self.num_classes_gestures, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
