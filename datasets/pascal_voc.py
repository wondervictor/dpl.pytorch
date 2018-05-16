# -*- coding: utf-8 -*-

"""
Description: Image Loader of PASCAL VOC
Code written by wondervictor, most code borrowed from Ross Girshick et. al
"""

import os
import torch
import pickle
import random
import subprocess
import numpy as np
import scipy.io as sio
import scipy.sparse
from PIL import Image
import prepare_dense_box
from voc_eval import voc_eval
from torch.utils.data import Dataset
import xml.etree.ElementTree as eTree
import torchvision.transforms as transforms

ROOT_DIR = ''
MATLAB = ''


class PASCALVOC(Dataset):

    def __init__(self, imageset, data_dir, img_size, roi_path, roi_type='dense_box', devkit=None):
        super(PASCALVOC, self).__init__()

        self._imageset = imageset  # 'trainval', 'val'
        self._data_path = data_dir
        self._classes = ('__backgroud__', 'aeroplane', 'bicycle', 'bird',
                         'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                         'person', 'pottedplant', 'sheep', 'sofa', 'train',
                         'tvmonitor')
        self.img_ext = '.jpg'
        self.img_size = img_size
        self._class_to_index = dict(list(zip(self._classes, range(len(self._classes)))))
        self._image_index = self._load_imageset_index()
        self.num_classes = len(self._classes)
        self._devkit_path = devkit if devkit is not None else 'VOCdevkit'
        self._anotations = dict([(x, self._load_anotation_from_index(x)) for x in self._image_index])
        self._labels = dict([(x, self._load_class_label_from_index(x)) for x in self._image_index])
        self.roi_path = roi_path
        self.roi_type = roi_type
        self.toTensor = transforms.ToTensor()
        self.resize = transforms.Resize(img_size)

        self._load_rois(roi_type=self.roi_type, roi_dir=self.roi_path)
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None}

    def __len__(self):
        return len(self._image_index)

    def _load_imageset_index(self):
        """ Load Image Index from File
        """
        imageset_file = os.path.join(self._data_path, 'ImageSets', 'Main', self._imageset, '.txt')
        assert os.path.exists(imageset_file), 'Path does not exists: {}'.format(imageset_file)

        with open(imageset_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, 'JPEGImages', index+self.img_ext)
        return image_path

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def _load_class_label_from_index(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = eTree.parse(filename)
        objs = tree.findall('object')
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        gt_classes = np.zeros(num_objs, dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            cls = self._class_to_index[obj.find('name').text.lower().strip()]
            gt_classes[ix] = cls

        real_label = np.zeros(self.num_classes).astype(np.float32)
        for label in gt_classes:
            real_label[label] = 1

        return {'labels': real_label}

    def _load_anotation_from_index(self, index):
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = eTree.parse(filename)
        objs = tree.findall('object')

        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros(num_objs, dtype=np.float32)

        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')

            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            cls = self._class_to_index[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
            "boxes": boxes,
            "gt_classes": gt_classes,
            "gt_overlaps": overlaps,
            "flipped": False,
            "seg_ares": seg_areas
        }

    def _get_voc_results_file_template(self):
        filename = '_det_' + self._imageset + '_{:s}.txt'
        path = os.path.join(
            self._devkit_path,
            'results',
            'VOC' + '2012',
            'Main',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            print('Writing {} VOC results file'.format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self._image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if len(dets) == 0:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir='output'):
        annopath = os.path.join(
            self._devkit_path,
            'VOC2012',
            'Annotations',
            '{:s}.xml')
        imagesetfile = os.path.join(
            self._devkit_path,
            'VOC2012',
            'ImageSets',
            'Main',
            self._imageset + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
                use_07_metric=False)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print('-----------------------------------------------------')
        print('Computing results with the official MATLAB eval code.')
        print('-----------------------------------------------------')
        path = os.path.join(ROOT_DIR, 'dataset',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
            .format(self._devkit_path, self._imageset, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        if self.config['matlab_eval']:
            self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def _create_patches(self, roi_dir):
        boxes = prepare_dense_box.prepare_dense_box(os.path.join(self._data_path, 'JPEGImages'), self._image_index,
                                                    os.path.join(roi_dir, "pascal_{}_box.pkl".format(self._imageset)))
        return boxes

    def _load_selective_search(self, roi_dir):

        filename = os.path.join(roi_dir, 'selective_search_data', 'voc_2012_' + self._imageset + '.mat')
        assert os.path.exists(filename), "Path: {} does not exists!".format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()
        box_list = {}
        for i in xrange(raw_data.shape[0]):
            box_list[self._image_index[i]] = raw_data[i][:, (1, 0, 3, 2)] - 1
        return box_list

    def _load_rois(self, roi_type, roi_dir):

        if roi_type == 'dense_box':
            self.rois = self._create_patches(roi_dir)
        elif roi_type == 'selective_search':
            self.rois = self._load_selective_search(roi_dir)

    def __getitem__(self, idx):

        img_name = self._image_index[idx]
        img_path = self.image_path_at(idx)
        img = Image.open(img_path)

        roi = self.rois[img_name]
        label = self._labels[img_name]

        w, h = img.size
        max_size = max(h, w)
        ratio = float(self.img_size) / float(max_size)
        w = int(w*ratio)
        h = int(h*ratio)
        img = img.resize((w, h))
        img = self.toTensor(img)
        roi = roi * ratio
        if len(roi) > 2000:
            roi = random.sample(roi, 2000)

        wrap_img = torch.zeros((3, self.img_size, self.img_size))
        wrap_img[:, 0:w, 0:h] = img

        return wrap_img, label, roi







