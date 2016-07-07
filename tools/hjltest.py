import random
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from datasets.factory import get_imdb
#from fast_rcnn.test import im_detect
from utils.timer import Timer
#import caffe
import argparse
import pprint
import numpy as np
import numpy.random as npr
import cv2
from sklearn import svm
import os, sys
from array import array
import cPickle as pickle

from PIL import Image, ImageDraw

w = np.loadtxt("/mnt/c/Users/T-HUNTEL/Source/Repos/fast-rcnn/tools/svmweights.txt")
b = np.loadtxt("/mnt/c/Users/T-HUNTEL/Source/Repos/fast-rcnn/tools/svmbias.txt")


imdb = get_imdb('voc_2007_test')

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(0, 2476):
    with open("/mnt/d/testfeat/{}.arr".format(i)) as inf:
        float_arr = array('d')
        float_arr.fromfile(inf, 2000*4096)
        feat = np.array(float_arr.tolist(), dtype='float').reshape((2000, 4096))

        # 2000 x 21 score matrix.
        scores = np.dot(feat, w) + b

        # compute argmax along column dimension to get class score for that box.
        labels = np.argmax(scores, axis=1)

        # chop off extra boxes we added to get to 2000
        labels = labels[0:min(2000, len(imdb.roidb[i]['boxes']))]
        

        # where are the ground-truth boxes for this image
        gt_inds = np.where(imdb.roidb[i]['gt_classes'] > 0)[0]
        if len(gt_inds) == 0:
            continue
        
        # what are the gt classes
        gt_classes = imdb.roidb[i]['gt_classes'][gt_inds]
        print "image {} has classes {}".format(i, gt_classes)
        print imdb.image_path_at(i)


        num_gt_right = len(np.where(labels[gt_inds] == gt_classes)[0])
        num_gt = len(gt_inds)
        print "{} / {} gt boxes classified correctly".format(num_gt_right, num_gt)

        non_bg_pred = np.where(labels > 0)[0]
        for b_idx in non_bg_pred:
            pred_class = labels[b_idx]
            if imdb.roidb[i]['gt_overlaps'][b_idx, pred_class] > 0.5:
                tp += 1
            else:
                # print "FP for box {} class {}: true overlap is {}".format(b_idx, pred_class, imdb.roidb[i]['gt_overlaps'][b_idx, pred_class])
                fp += 1

        bg_pred = np.where(labels == 0)[0]
        for b_idx in bg_pred:
            # if the box actually has > 0.5 overlap with some class
            # and we said 0, it's a false negative
            if len(np.where(imdb.roidb[i]['gt_overlaps'][b_idx, :].toarray() > 0.5)[0]) > 0:
                fn += 1
            else:
                tn += 1

                #sys.exit()

        """
        im = Image.open(imdb.image_path_at(i))
        for gt_idx in gt_inds:
            box = imdb.roidb[i]['boxes'][gt_idx]
            bbox = [(box[0], box[1]), (box[2], box[3])]
            draw = ImageDraw.Draw(im)
            draw.rectangle(bbox)
            del draw

        for p_idx in pos_loc:
            box = imdb.roidb[i]['boxes'][p_idx]
            bbox = [(box[0], box[1]), (box[2], box[3])]
            draw = ImageDraw.Draw(im)
            draw.rectangle(bbox, outline=128)
            del draw

        im.save("output.png")
        """

print "------overall stats-----\ntp: {}; fn: {}; tn: {} fp: {}\npos_acc: {}; neg_acc: {}; overall: {}".format(tp, fn, tn, fp, float(tp)/float(tp + fn), float(tn)/float(tn + fp), float(tp + tn) / float(tp + tn + fp + fn))
