# -*- coding: utf-8 -*-

"""
Description: Create Dataset
Author: wondervictor
"""

import os
import torch
import numpy as np
import torch.utils.data.dataset as datasets
import xml.etree.ElementTree as eTree
from PIL import Image
import torchvision.transforms as transforms
import prepare_dense_box

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])


class PASCAL(datasets.Dataset):

    def __init__(self, data_path, transform=None, train=True, name='VOC2012'):
        super(PASCAL, self).__init__()
        self.data_path = data_path
        self.train = train
        self._pre_load()
        self._init_classes()
        self._create_patches()
        print("----- Finish Creating Patches ------")
        self.transfrom = transform

    def __getitem__(self, index):
        img_name = self.image_list[index]
        img_path = os.path.join(self.data_path, 'JPEGImages', "{}.jpg".format(img_name))
        img = Image.open(img_path)
        if self.transfrom is None:
            img = transform(img)
        else:
            img = self.transfrom(img)
        label = self.labels[img_name]
        label = torch.FloatTensor(self._convert_label(label))
        box = self.boxes[img_name]
        return img, label, box

    def _convert_label(self, lbl):
        label = np.zeros(20, dtype=np.int32)
        for l in lbl:
            label[self.classes_map[l]] = 1
        return label

    def _pre_load(self):
        if self.train:
            trainval = 'train.txt'
        else:
            trainval = 'val.txt'
        annotation_path = os.path.join(self.data_path, 'ImageSets', 'Main', trainval)
        with open(annotation_path, 'r') as f:
            image_list = f.readlines()
        image_list = map(lambda x: x.rstrip('\n\r'), image_list)
        self.image_list = image_list
        label_dir = os.path.join(self.data_path, 'Annotations')
        labels = {}
        for img_id in image_list:
            xml_file = os.path.join(label_dir, '{}.xml'.format(img_id))
            tree = eTree.parse(xml_file)
            objs = tree.findall('object')
            label = set()
            for idx, obj in enumerate(objs):
                cls = obj.find('name').text.lower().strip()
                label.add(cls)
            labels[img_id] = label
        self.labels = labels
        print("----------- Finished Dataset Preloading -----------")

    def _create_patches(self):
        phase = 'train' if self.train else 'val'
        boxes = prepare_dense_box.prepare_dense_box(self.data_path, self.image_list, "pascal_{}_box.pkl".format(phase))
        self.boxes = boxes

    def __len__(self):
        return len(self.image_list)

    def _init_classes(self):
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        self.classes_map = {'sheep': 16, 'horse': 12, 'bicycle': 1, 'aeroplane': 0, 'cow': 9, 'sofa': 17, 'bus': 5,
                            'dog': 11, 'cat': 7, 'person': 14, 'train': 18, 'diningtable': 10, 'bottle': 4, 'car': 6,
                            'pottedplant': 15, 'tvmonitor': 19, 'chair': 8, 'bird': 2, 'boat': 3, 'motorbike': 13}


class COCO(datasets.Dataset):
    def __init__(self, data_path):
        super(COCO, self).__init__()
        self.data_path = data_path

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
