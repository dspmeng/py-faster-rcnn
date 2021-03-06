# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
#import datasets.ds_utils as ds_utils
#import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
#import scipy.io as sio
import utils.cython_bbox
import cPickle
#import subprocess
#import uuid
import cv2
from fast_rcnn.config import cfg

class gehler(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'gehler' + '_' + image_set)
        self._image_set = image_set
        self._data_path = os.path.join(cfg.DATA_DIR, 'gehler')
        self._classes = ('__background__', # always index 0
                         'colorchecker',
                         'cb11', 'cb12', 'cb13', 'cb14', 'cb15', 'cb16',
                         'cb21', 'cb22', 'cb23', 'cb24', 'cb25', 'cb26',
                         'cb31', 'cb32', 'cb33', 'cb34', 'cb35', 'cb36',
                         'cb41', 'cb42', 'cb43', 'cb44', 'cb45', 'cb46')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.gt_roidb

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'images', index)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # cfg.DATA_DIR/gehler/trainval.txt
        image_set_file = os.path.join(self._data_path,
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_labels(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _clamp(self, x, low, high):
        return max(min(x, high), low)

    def _load_labels(self, index):
        """
        Load image and bounding boxes info from coordinate files under
        cfg.DATA_DIR/gehler/labels.
        """
        filename = os.path.join(self._data_path, 'labels', index.split('.')[0] + '_macbeth.txt')
        assert os.path.exists(filename), \
                'Path does not exist: {}'.format(filename)
        """
        Gehler coordinate file has following format:
        width              height
        cc-top-left-x      cc-top-left-y
        cc-top-right-x     cc-top-right-y
        cc-bottom-left-x   cc-bottom-left-y
        cc-bottom-right-x  cc-bottom-right-y
        cb-top-left-x      cb-top-left-y
        cb-top-right-x     cb-top-right-y
        cb-bottom-right-x  cb-bottom-right-y
        cb-bottom-left-x   cb-bottom-left-y
        ...
        """
        img = cv2.imread(self.image_path_from_index(index))
        with open(filename) as f:
            width, height = [float(i) for i in f.readline().split()]
            vertices = [map(float, line.split()) for line in f.readlines()]
            vertices[2], vertices[3] = vertices[3], vertices[2]
            vertices = np.array(vertices)
            vertices[:,0] = [self._clamp(float(x) * img.shape[1] / width, 0, img.shape[1] - 1)
                             for x in vertices[:,0]]
            vertices[:,1] = [self._clamp(float(y) * img.shape[0] / height, 0, img.shape[0] - 1)
                             for y in vertices[:,1]]
            vertices.astype(np.uint16)

        num_objs = vertices.shape[0] / 4
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for gehler is just the color checker area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        crop = np.zeros((4), dtype=np.int32)

        vis = False
        boxes[0, :] = np.concatenate([vertices[0:4,:].min(0), vertices[0:4,:].max(0)])
        if vis:
            cv2.rectangle(img, (boxes[0,0], boxes[0,1]), (boxes[0,2], boxes[0,3]), (0, 255, 0), 1)
        gt_classes[0] = 1
        overlaps[0, 1] = 1.0
        seg_areas[0] = (boxes[0, 2] - boxes[0, 0] + 1) * (boxes[0, 3] - boxes[0, 1] + 1)
        for i in range(1, num_objs):
            boxes[i, :] = np.concatenate([vertices[4*i:4*i+4,:].min(0), vertices[4*i:4*i+4,:].max(0)])
            cls = i + 1
            gt_classes[i] = cls
            overlaps[i, cls] = 1.0
            seg_areas[i] = (boxes[i, 2] - boxes[i, 0] + 1) * (boxes[i, 3] - boxes[i, 1] + 1)
            if vis:
                cv2.rectangle(img, (boxes[i,0], boxes[i,1]), (boxes[i,2], boxes[i,3]), (0, 255, 0), 1)
        if vis:
            cv2.imshow('image',img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'noisy' : False,
                'cropped' : False,
                'crop' : crop,
                'seg_areas' : seg_areas}

    def evaluate_detections(self, all_boxes, output_dir):
        pass

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.gehler import gehler
    d = gehler('trainval')
    res = d.roidb
    from IPython import embed; embed()
