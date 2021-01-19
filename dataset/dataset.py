# /usr/bin/python
# -*- coding:utf-8 -*-
"""
    @description: dataset build
    @detail:
    @copyright: xxx
    @author: Seven Hsu
    @e-mail: xxx
    @date: xxx
"""
import os
import cv2
import torch
import random
import numpy as np
import os.path as osp
from torchvision import transforms
from torch.utils.data import Dataset
from utils.utils import gaussian_radius, draw_msra_gaussian, shrink_poly


class OCRDataset(Dataset):
    def __init__(self, data_root, gt_root, list_file, input_size, mode='train', label_table=None, max_seq=None):
        self.img_root = data_root
        self.gt_root = gt_root
        self.list_file = list_file
        self.input_size = input_size
        self.mode = mode
        self.label_table = label_table
        self.max_seq = max_seq
        self.num_classes = len(label_table)
        self.shrink_rate = 0.5
        self.gaussian_thresh = 0.3
        self.shrink_mode = 'poly'

        with open(list_file, 'r') as f:
            self.img_files = f.read().splitlines()
        self.label_map = dict(zip(self.label_table, [x for x in range(self.num_classes)]))

        self.transforms = transforms.Compose(
            [transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1, hue=0.1),
             transforms.ToTensor()])

    def __getitem__(self, idx):
        img_path = osp.join(self.img_root, self.img_files[idx])
        if not osp.exists(img_path):
            self.__getitem__(random.randint(0, len(self.img_files) - 1))
        img = cv2.imread(img_path)

        # dilate or erode the org image randomly
        if random.random() < 0.15 and self.mode == 'train':
            if random.random() < 0.5:
                img = cv2.dilate(img, np.ones((3, 3)))
            else:
                img = cv2.erode(img, np.ones((3, 3)))
        # blur image randomly
        if random.random() < 0.15 and self.mode == 'train':
            if random.random() < 0.5:
                img = cv2.blur(img, (3, 3))
            else:
                scale_ratio = random.uniform(0.3, 0.8)
                raw_h, raw_w, _ = img.shape
                new_h, new_w = int(raw_h * scale_ratio), int(raw_w * scale_ratio)
                img = cv2.resize(img, (new_w, new_w))
                img = cv2.resize(img, (raw_w, raw_h))
        # load annotation
        anno_path = osp.join(self.gt_root, self.img_files[idx].split('.')[0] + '.txt')
        with open(anno_path, 'r') as f:
            anno = f.read().splitlines()

        anno = anno[0].split(',')
        anno = np.asarray(anno).reshape(-1, 9)

        chars_seg, order_seg, pos_seg, text = self._gen_target(anno, img.shape[:-1])

        img = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        img = np.transpose(img, [2, 0, 1])
        img = torch.as_tensor(img / 255., dtype=torch.float32)

        if self.mode != 'test':
            chars_seg = torch.as_tensor(chars_seg, dtype=torch.uint8)
            order_seg = torch.as_tensor(order_seg, dtype=torch.uint8)
            pos_seg = torch.as_tensor(pos_seg, dtype=torch.float32)

            return img, chars_seg, order_seg, pos_seg
        else:
            return img, text

    def _gen_target(self, anno, img_shape):
        if self.mode == 'test':
            return None, None, None, anno[0, -1]
        h, w = self.input_size
        im_scale = (w / img_shape[1], h / img_shape[0])
        chars_seg = np.zeros(self.input_size, dtype=np.uint8)
        order_map = np.zeros(self.input_size, dtype=np.uint8)
        all_gaus_map = np.zeros((self.max_seq, h, w), dtype=np.float32)

        for k, char_anno in enumerate(anno):
            char_label = char_anno[-1]
            if len(char_label) > 1:
                continue
            points = np.array([float(x) for x in char_anno[:-1]])
            points = points.reshape(-1, 2) * im_scale
            shrunk_poly_bbox = shrink_poly(points, self.shrink_rate, self.shrink_mode)
            cv2.fillPoly(chars_seg, [shrunk_poly_bbox], self.label_map[char_label])

            char_w, char_h = points.max(0) - points.min(0)
            center = ((points.max(0) + points.min(0)) / 2).astype(np.int)
            radius = gaussian_radius((char_h, char_w))
            gaussian_map = np.zeros(self.input_size, dtype=np.float32)
            gaussian_map = draw_msra_gaussian(gaussian_map, center, radius)
            all_gaus_map[k - 1] = gaussian_map
            max_gaus_val = np.max(gaussian_map)
            idx = np.where((gaussian_map / max_gaus_val) > self.gaussian_thresh)
            order_map[idx[0], idx[1]] = k
        pos_seg = all_gaus_map.max(0)

        return chars_seg, order_map, pos_seg, None

    def __len__(self):
        return len(self.img_files)
