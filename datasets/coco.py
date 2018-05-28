# -*- coding: utf-8 -*-

"""
description: COCO Dataset with Torch
author: wondervictor
Part of the code by Ross Girshick and Xinlei Chen
"""

import sys
import os
import pickle
import json
import uuid
import numpy as np
import scipy.sparse
import os.path as osp
from PIL import Image
import scipy.io as sio

# COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as COCOmask

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    assert (x1 >= 0).all()
    assert (y1 >= 0).all()
    assert (x2 >= x1).all()
    assert (y2 >= y1).all()
    assert (x2 < width).all()
    assert (y2 < height).all()


class COCODataset(Dataset):

    def __init__(self, data_dir, img_size, imageset, year):
        super(COCODataset, self).__init__()

        self.config = {"use_salt": True,
                       "cleanup": True}

        self.data_dir = data_dir
        self._imageset = imageset
        self._year = year
        self._img_size = img_size

        self._coco = COCO(annotation_file=self._get_annotation_file())
        self._categories = self._coco.loadCats(self._coco.getCatIds())
        self.classes = ['__background__'] + [c['name'] for c in self._categories]
        self.num_classes = len(self.classes)
        self._class_to_index = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_cat_id = dict(list(zip([c['name'] for c in self._categories], self._coco.getCatIds())))
        self._image_index = self._load_image_set_index()

        self._view_map = {
          'minival2017': 'val2017',  # 5k val2014 subset
          'test-dev2017': 'test2017',
          'capval2017': 'val2017',
          'captest2017': 'val2017'
        }
        coco_name = imageset + year
        self._data_name = self._view_map[coco_name] if coco_name in self._view_map else coco_name
        self._gt_splits = ['train', 'val', 'minval']
        self.annotations = dict(zip(self._image_index, [self._load_coco_annotation(ind) for ind in self._image_index]))
        self.toTensor = transforms.ToTensor()

    def _load_image_set_index(self):
        img_ids = self._coco.getImgIds()
        return img_ids

    def image_path_at(self, i):
        return self.image_path_from_index(self.image_id_at(i))

    def image_id_at(self, i):
        return self._image_index[i]

    def image_path_from_index(self, index):
        """ Get Image Path by Index"""
        # index = 119993
        # filename = 000000119993.jpg
        # filepath = 'train2017/000000119993.jpg'
        img_name = str(index).zfill(12)+'.jpg'
        img_path = osp.join(self.data_dir, self._data_name, img_name)
        return img_path

    def _get_annotation_file(self):
        """ get COCO annotation file """
        prefix = 'instances' if self._imageset.find('test') == -1 \
            else 'image_info'
        return osp.join(self.data_dir, 'annotations', prefix + '_' + self._imageset + self._year + '.json')

    def _load_coco_annotation(self, index):
        """ Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.

        Args：
            index: int, image index
        Return:

        """
        im_ann = self._coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._coco.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._coco.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        seg_areas = np.zeros(num_objs, dtype=np.float32)

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_index[cls])
                                         for cls in self.classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                overlaps[ix, :] = -1.0
            else:
                overlaps[ix, cls] = 1.0

        validate_boxes(boxes, width=width, height=height)
        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'width': width,
                'height': height,
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _print_detection_eval_metrics(self, coco_eval):
        iou_thresh_min = 0.5
        iou_thresh_max = 0.95

        def _get_thresh_ind(_coco_eval, thresh):
            ind = np.where((_coco_eval.params.iouThrs > thresh-1e-5) &
                           (_coco_eval.params.iouThrs < thresh+1e-5))[0][0]
            iou_thresh = _coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thresh, thresh)
            return ind

        ind_lo = _get_thresh_ind(coco_eval, iou_thresh_min)
        ind_hi = _get_thresh_ind(coco_eval, iou_thresh_max)

        # precision: [iou, recall, cls, area range, max_dets]
        # area range index 0: all ranges
        # max dets index 2: 100 per image
        precision = coco_eval.eval['precision'][ind_lo:(ind_hi+1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(iou_thresh_min, iou_thresh_max))
        print('{:.1f}'.format(100 * ap_default))

        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi+1), :, cls_ind-1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{:.1f}'.format(100 * ap))
        print('~~~~ Summary Metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._coco.loadRes(res_file)
        coco_eval = COCOeval(self._coco, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = osp.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self._image_index):
            dets = boxes[im_ind].astype(np.float)
            if len(dets) == 0:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
                                                             self.num_classes - 1))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = osp.join(output_dir, ('detections_' +
                                         self._imageset +
                                         self._year +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self._imageset.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def __len__(self):
        return len(self._image_index)

    def __getitem__(self, idx):
        ind = self._image_index[idx]
        img_path = self.image_path_from_index(ind)
        anno = self.annotations[ind]

        boxes = anno['boxes']
        gt_classes = anno['gt_classes']

        img = Image.open(img_path)

        h, w = img.size
        max_size = max(h, w)
        ratio = float(self._img_size) / float(max_size)
        w = int(w*ratio)
        h = int(h*ratio)
        img = img.resize((h, w))
        img = self.toTensor(img)
        boxes = boxes * ratio

        wrap_img = torch.zeros((3, self._img_size, self._img_size))
        wrap_img[:, 0:w, 0:h] = img
        if self._imageset == 'test':
            return wrap_img, gt_classes, boxes, np.array([w, h], dtype=np.float32)
        else:
            return wrap_img, gt_classes, boxes, np.array([w, h], dtype=np.float32)
