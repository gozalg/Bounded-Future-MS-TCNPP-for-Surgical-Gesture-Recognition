import torch
import torch.utils.data as data
import torchvision
from utils.transforms import Stack, ToTorchFormatTensor

from PIL import Image
import os
import numpy as np
from numpy.random import randint
import random
import math
import cv2
from skimage.util import random_noise


class Gesture2dTrainSet(data.Dataset):
    def __init__(self,list_of_train_examples, root_path, transcriptions_dir, gesture_ids,
                 samoling_factor=6,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_side",
                 transform=None, normalize=None, epoch_size = 50,debag=False):
        self.list_of_train_examples =list_of_train_examples
        self.debag = debag
        self.root_path = root_path
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.transform = transform
        self.normalize = normalize
        self.epoch_size =epoch_size
        self.samoling_factor =samoling_factor
        self.frames_indces_by_gesture ={}

        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.labels_data = {}
        self._parse_list_files()
        self._sort_frames_by_gesture()


    def _parse_list_files(self):
        # depands only on csv files of splits directory
        videos = self.list_of_train_examples
        if self.debag:
            videos = [videos[0]]
        for video in videos:
            video_id = video[:-4]

            file_ptr = open(os.path.join(self.transcriptions_dir, video.split('.')[0] + '.txt'), 'r')
            gt_source = file_ptr.read().split('\n')[:-1]
            gestures  = self.pars_ground_truth(gt_source)



            _last_rgb_frame =0
            for file in os.listdir(os.path.join(self.root_path, video_id + self.video_suffix)):
                filename = os.fsdecode(file)
                if int(filename[4:9]) > _last_rgb_frame:
                    _last_rgb_frame = int(filename[4:9])

            if len(gestures) >= _last_rgb_frame:
                gestures =gestures[:_last_rgb_frame]
            else:
                padding_elements = [gestures[-1]] * (_last_rgb_frame - len(gestures))
                gestures = gestures + padding_elements

            assert (len(gestures) == _last_rgb_frame)


            self.labels_data[video_id] = gestures



    def pars_ground_truth(self,gt_source):
        contant =[]
        for line in gt_source:
            info = line.split()
            line_contant = [info[2]] * (int(info[1])-int(info[0]) +1)
            contant = contant + line_contant
        return contant


    def _sort_frames_by_gesture(self):
        for experiment_name in self.labels_data:
            self.frames_indces_by_gesture[experiment_name] = {}
            for gesture in self.gesture_ids:
                self.frames_indces_by_gesture[experiment_name][gesture] =[]
            for i, gesture in enumerate(self.labels_data[experiment_name]):
                self.frames_indces_by_gesture[experiment_name][gesture].append(i+1)


    def _preload_images(self, video_id,_last_rgb_frame):
        print("Preloading images from video {}...".format(video_id))
        images = []
        img_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        for idx in range(1,_last_rgb_frame + 1,self.samoling_factor):
            imgs = self._load_image(img_dir, idx)
            images.extend(imgs)
        self.image_data[video_id] = images


    def _load_image(self, directory, idx):
        img = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')
        return [img]



    def __getitem__(self, index):
        video_select = randint(0,len(self.labels_data))
        video_name = list(self.labels_data.keys())[video_select]
        gesture_select = randint(0,len(self.gesture_ids))
        gesture = list(self.gesture_ids)[gesture_select]
        target = gesture_select
        num_of_frames_in_video = len(self.frames_indces_by_gesture[video_name][gesture])
        frame_code_select = randint(0,num_of_frames_in_video)
        frame_select = self.frames_indces_by_gesture[video_name][gesture][frame_code_select]
        data = self.get_frame(video_name, frame_select)

        #target_individual = torch.tensor(int(label[1]))

        return data, target




    def get_frame(self, video_id, idx):
        data = list()
        if idx < 1:
            raise ValueError
        mg_dir = os.path.join(self.root_path, video_id + self.video_suffix)
        img=self._load_image(mg_dir,idx)
        img =img[0]
        data.append(img)
        data = rotate_snippet(data,0.5)
        data = self.transform(data)
        data = [torchvision.transforms.ToTensor()(img) for img in data]
        data = data[0]
        data = self.normalize(data)
        return data


    def __len__(self):
        return self.epoch_size





def rotate_snippet(snippet,max_angle):
    preform = random.uniform(0,1)
    if preform > 0.5:
        new_snippet =[]
        angle = random.uniform(-max_angle,max_angle)
        for img in snippet:
            new_img = rotate_img(img, angle)
            new_snippet.append(new_img)
        return new_snippet
    else:
        return snippet


def rotate_img(image, angle):
    image = np.array(image)
    num_rows, num_cols = (image.shape[:2])
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))
    image = Image.fromarray(image)

    return image


def Add_Gaussian_Noise_to_snippet(snippet):
    preform = random.uniform(0,1)
    if preform > 0.5:
        new_snippet =[]
        sigma = random.uniform(0,0.08)
        for img in snippet:
            new_img = Add_Gaussian_Noise(img,sigma)
            new_snippet.append(new_img)
        return new_snippet
    else:
        return snippet

class Sequential2DTestGestureDataSet(data.Dataset):

    def __init__(self, root_path, video_id, transcriptions_dir, gesture_ids,
                 snippet_length=16,sampling_step = 6,
                 image_tmpl='img_{:05d}.jpg', video_suffix="_side",
                 return_3D_tensor=True, return_dense_labels=True,
                 transform=None, normalize=None,preload= False):

        self.root_path = root_path
        self.video_name = video_id[:-4]
        self.video_id = video_id[:-4]
        self.transcriptions_dir = transcriptions_dir
        self.gesture_ids = gesture_ids
        self.snippet_length = snippet_length
        self.sampling_step = sampling_step
        self.image_tmpl = image_tmpl
        self.video_suffix = video_suffix
        self.return_3D_tensor = return_3D_tensor
        self.return_dense_labels = return_dense_labels #
        self.transform = transform
        self.normalize = normalize
        self.preload =preload

        self.gesture_sequence_per_video = {}
        self.image_data = {}
        self.frame_num_data = {}
        self.labels_data = {}
        self._parse_list_files(self.video_id)

    def _parse_list_files(self, video_name):
        # depands only on csv files of splits directory
        video_id = video_name

        gestures_file = os.path.join(self.transcriptions_dir, video_id + ".txt")
        gestures = [[int(x.strip().split(' ')[0]), int(x.strip().split(' ')[1]), x.strip().split(' ')[2]]
                    for x in open(gestures_file)]
        # [start_frame, end_frame, gesture_id]

        _initial_labeled_frame = gestures[0][0]
        _final_labaled_frame = gestures[-1][1]

        _last_rgb_num = 0
        for file in os.listdir(os.path.join(self.root_path, video_id + self.video_suffix)):
            filename = os.fsdecode(file)
            if int(filename[4:9]) > _last_rgb_num:
                _last_rgb_num = int(filename[4:9])

        _last_rgb_frame = os.path.join(self.root_path, video_id + self.video_suffix,
                                       'img_{:05d}.jpg'.format(_last_rgb_num))


        if _final_labaled_frame > _last_rgb_num:
            _final_labaled_frame = _last_rgb_num

        self.frame_num_data[video_id] = list(range(1,_final_labaled_frame + 1,self.sampling_step))
        self._generate_labels_list(video_id,gestures)
        if self.preload:
            self._preload_images(video_id)
            assert len(self.image_data[video_id]) == len(self.labels_data[video_id])



    def _generate_labels_list(self,video_id,gestures):
        labels_list =[]

        for frame_num in self.frame_num_data[self.video_name]:
            for gesture in gestures:
                if frame_num >= gesture[0] and frame_num <= gesture[1]:
                    labels_list.append(self.gesture_ids.index(gesture[2]))
                    break
        self.labels_data[video_id] = labels_list


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
        return [img]

    def __getitem__(self, index):
        frame_list = [index]
        target =torch.tensor(self.labels_data[self.video_id][index], dtype=torch.long)
        data = self.get_snippet(self.video_name,index)

        return data, target


    def get_snippet(self, video_id, idx):
        snippet = list()
        _idx = max(idx, 0)  # padding if required
        if self.preload:
            img = self.image_data[video_id][_idx]
        else:
            mg_dir = os.path.join(self.root_path, video_id + self.video_suffix)
            img=self._load_image(mg_dir,self.frame_num_data[self.video_name][idx])
            img =img[0]


        snippet.append(img)
        # snippet = rotate_snippet(snippet,0.5)
        # Add_Gaussian_Noise_to_snippet(snippet)
        snippet = self.transform(snippet)
        snippet = [torchvision.transforms.ToTensor()(img) for img in snippet]
        snippet = snippet[0]
        snippet = self.normalize(snippet)
        data = snippet
        return data


    def __len__(self):
        return (len(self.labels_data[self.video_name])) -(self.snippet_length -1)


def Add_Gaussian_Noise(image,sigma):
    image = np.array(image)
    noisyRandom = random_noise(image, var=sigma ** 2)
    im = Image.fromarray((noisyRandom * 255).astype(np.uint8))
    return im






