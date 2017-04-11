import os
import re
import sys
from matplotlib import pyplot as plt

# Iteration 26600, loss = 0.317481
#     Train net output #0: loss_bbox = 0.0679093 (* 1 = 0.0679093 loss)
#     Train net output #1: loss_cls = 0.0295109 (* 1 = 0.0295109 loss)
#     Train net output #2: rpn_cls_loss = 0.000879483 (* 1 = 0.000879483 loss)
#     Train net output #3: rpn_loss_bbox = 0.454861 (* 1 = 0.454861 loss)
regex_loss = re.compile(r'Iteration.*loss = (\d+\.\d+)')
regex_loss_bbox = re.compile(r'\sloss_bbox = (\d+\.\d+)')
regex_loss_cls = re.compile(r'loss_cls = (\d+\.\d+)')
regex_rpn_cls_loss = re.compile(r'rpn_cls_loss = (\d+\.\d+)')
regex_rpn_loss_bbox = re.compile(r'rpn_loss_bbox = (\d+\.\d+)')

def vis(log_file):
    loss = []
    loss_bbox = []
    loss_cls = []
    rpn_cls_loss = []
    rpn_loss_bbox = []
    with open(log_file) as log:
        for line in log:
            match = regex_loss.search(line)
            if match:
                loss.append(match.group(1))
                continue
            match = regex_loss_bbox.search(line)
            if match:
                loss_bbox.append(match.group(1))
                continue
            match = regex_loss_cls.search(line)
            if match:
                loss_cls.append(match.group(1))
                continue
            match = regex_rpn_cls_loss.search(line)
            if match:
                rpn_cls_loss.append(match.group(1))
                continue
            match = regex_rpn_loss_bbox.search(line)
            if match:
                rpn_loss_bbox.append(match.group(1))
        plt.figure(1)
        plt.subplot(221)
        plt.plot(loss_bbox)
        plt.title('loss_bbox')
        plt.subplot(222)
        plt.plot(loss_cls)
        plt.title('loss_cls')
        plt.subplot(223)
        plt.plot(rpn_loss_bbox)
        plt.title('rpn_loss_bbox')
        plt.subplot(224)
        plt.plot(rpn_cls_loss)
        plt.title('rpn_cls_loss')
        plt.figure(2)
        plt.plot(loss)
        plt.show()

if __name__ == '__main__':
    vis(sys.argv[1])
