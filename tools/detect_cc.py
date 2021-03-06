#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os, sys, cv2
import argparse
import rpn.proposal_layer as rpnpl
import pdb

os.environ['GLOG_minloglevel'] = '1'
import caffe

CLASSES = ('__background__',
           'colorchecker',
           'cb11', 'cb12', 'cb13', 'cb14', 'cb15', 'cb16',
           'cb21', 'cb22', 'cb23', 'cb24', 'cb25', 'cb26',
           'cb31', 'cb32', 'cb33', 'cb34', 'cb35', 'cb36',
           'cb41', 'cb42', 'cb43', 'cb44', 'cb45', 'cb46')

def vis_detections(fig, ax, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    #rpnpl.im_name = image_name

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    CONF_THRESH = 0.2
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        scs = dets[:, 4]
        keep = np.argsort(scs, -1)
        dets = dets[keep[-1:], :]
        vis_detections(fig, ax, cls, dets, thresh=CONF_THRESH)
    ax.set_title(('{}').format(image_name), fontsize=14)
    fig.savefig('result_' + os.path.basename(image_name))
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument("model")
    parser.add_argument("image")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = 'models/gehler/VGG16/colorchecker/test.prototxt'
    caffemodel = args.model

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./experiments/scripts/colorchecker.sh'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = [
        'data/gehler/videos/Colorchecker_1.avi-001.png',
        'data/gehler/videos/Colorchecker_1.avi-002.png',
        'data/gehler/videos/Colorchecker_1.avi-003.png',
        'data/gehler/videos/Colorchecker_1.avi-004.png',
        'data/gehler/videos/Colorchecker_1.avi-005.png',
        'data/gehler/videos/Colorchecker_1.avi-006.png',
        'data/gehler/videos/Colorchecker_1.avi-007.png',
        'data/gehler/videos/Colorchecker_1.avi-008.png',
        'data/gehler/videos/Colorchecker_1.avi-009.png',
        'data/gehler/videos/Colorchecker_1.avi-010.png',
        'data/gehler/videos/Colorchecker_1.avi-011.png',
        'data/gehler/videos/Colorchecker_1.avi-012.png',
        'data/gehler/videos/Colorchecker_1.avi-013.png',
        'data/gehler/videos/Colorchecker_1.avi-014.png',
        'data/gehler/videos/Colorchecker_1.avi-015.png',
        'data/gehler/videos/Colorchecker_1.avi-016.png',
        'data/gehler/videos/Colorchecker_1.avi-017.png',
        'data/gehler/videos/Colorchecker_1.avi-018.png',
        'data/gehler/videos/Colorchecker.avi-001.png',
        'data/gehler/videos/Colorchecker.avi-002.png',
        'data/gehler/videos/Colorchecker.avi-003.png',
        'data/gehler/videos/Colorchecker.avi-004.png',
        'data/gehler/videos/Colorchecker.avi-005.png',
        'data/gehler/videos/Colorchecker.avi-006.png',
        'data/gehler/videos/Colorchecker.avi-007.png',
        'data/gehler/videos/Colorchecker.avi-008.png',
        'data/gehler/videos/Colorchecker.avi-009.png',
        'data/gehler/videos/Colorchecker.avi-010.png',
        'data/gehler/videos/Colorchecker.avi-011.png',
        'data/gehler/videos/Colorchecker.avi-012.png',
        'data/gehler/videos/Colorchecker.avi-013.png',
        'data/gehler/videos/Colorchecker.avi-014.png',
        'data/gehler/videos/Colorchecker.avi-015.png',
        'data/gehler/videos/Colorchecker.avi-016.png',
        'data/gehler/videos/Colorchecker.avi-017.png',
        'data/gehler/videos/Colorchecker.avi-018.png'
    ]
    im_names.append(args.image)
    for img in im_names:
        demo(net, img)

    plt.show()
