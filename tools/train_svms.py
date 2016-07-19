#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Train post-hoc SVMs using the algorithm and hyper-parameters from
traditional R-CNN.
"""
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
from scipy import io
import cPickle as pickle
from scipy.sparse import csr_matrix


LINE_OFFSETS = []
SAMPLE_MAP = []

#FEATMAT = np.zeros((2000, 4096))
#global FEATMAT

# todo: (hjl) rewrite to save on initialization time



class DummyNet(object):
    def __init__(self, dim, imdb):
        self.params = {
            "cls_score": [np.zeros((dim, imdb.num_classes)), np.zeros((1, imdb.num_classes))]
        }



#############################################################
################## DIFFERENT IM_LOAD methods ################
#############################################################
def im_detect4(net, im_idx, cand_box_idx, doProduct=True):
    imfeat = None
    inf = open("/mnt/d/samppkl/{}.pkl".format(im_idx), 'rb')
    sparse = pickle.load(inf)
    imfeat = sparse.toarray()

    scores = None
    if doProduct:
        scores = np.dot(imfeat, net.params['cls_score'][0]) + net.params['cls_score'][1]

    # scores should be #boxes x classes
    return scores, cand_box_idx, imfeat


def im_detect3(net, im_idx, cand_box_idx, doProduct=True):
    imfeat = None
    sparse = io.mmread("/mnt/d/sampmm/{}.mm.mtx".format(im_idx))
    # imfeat = sparse.toarray()
    imfeat = csr_matrix(sparse)

    scores = None
    if doProduct:
        scores = np.dot(imfeat, net.params['cls_score'][0]) + net.params['cls_score'][1]

    # scores should be #boxes x classes
    return scores, cand_box_idx, imfeat


def im_detect2(net, im_idx, cand_box_idx, doProduct=True):
    imfeat = None

    with open("/mnt/c/sampfeat/{}.arr".format(im_idx)) as inf:
        float_arr = array('d')
        float_arr.fromfile(inf, 4096*2000)
        imfeat = np.array(float_arr.tolist(), dtype='float').reshape((2000, 4096))

    if isinstance(cand_box_idx, list):
        imfeat = imfeat[cand_box_idx, :]
    elif isinstance(cand_box_idx, int):
        imfeat = imfeat[0:cand_box_idx, :]

    # compute dot product of svm weight matrix with features
    # and add bias term. 

    scores = None
    if doProduct:
        scores = np.dot(imfeat, net.params['cls_score'][0]) + net.params['cls_score'][1]

    # scores should be #boxes x classes
    return scores, cand_box_idx, imfeat


def im_detect(net, im_idx, cand_box_idx, doProduct=True):
    imfeat = None

    global LINE_OFFSETS

    samp_idx = SAMPLE_MAP.index(im_idx)
    print("reading {}".format(samp_idx))

    # read the features for this image from the cntk
    # output file.
    count = 0
    imfeat = np.zeros((2000, 4096))

    first_line = 2000*samp_idx
    offset = LINE_OFFSETS[first_line]
    size = LINE_OFFSETS[first_line + 1999] - offset

    with open("/mnt/d/vocsamp.h2.y") as fp:
        fp.seek(offset)
        lines = fp.readlines(size)

    fh = open('test.txt', 'w')
    for i,line in enumerate(lines):
        fh.write(line)
        imfeat[i, :] = np.fromstring(line, dtype=float, sep=' ')
    fh.close()
        
    print("done read of {}".format(samp_idx))
    
    # todo: reshape imfeat into a features x candidates
    # matrix.
    #imfeat = np.fromstring(imfeat, dtype=float, sep=' ')
    # imfeat = np.reshape(4096, 2000)

    # if we get a list (e.g. of ground truth box indices) only keep
    # around those corresponding features. if we get an int, chop off
    # the rest of the stuff since that int is the number of boxes we
    # have for this image.
    if isinstance(cand_box_idx, list):
        imfeat = imfeat[cand_box_idx, :]
    elif isinstance(cand_box_idx, int):
        imfeat = imfeat[0:cand_box_idx, :]

    # compute dot product of svm weight matrix with features
    # and add bias term. 

    scores = None
    if doProduct:
        scores = np.dot(imfeat, net.params['cls_score'][0]) + net.params['cls_score'][1]

    # scores should be #boxes x classes
    return scores, cand_box_idx, imfeat

#############################################################
################## END IM_LOAD methods ######################
#############################################################


class SVMTrainer(object):
    """
    Trains post-hoc detection SVMs for all classes using the algorithm
    and hyper-parameters of traditional R-CNN.
    """

    def __init__(self, net, imdb):
        self.imdb = imdb
        self.net = net
        self.layer = 'fc7'
        self.hard_thresh = -1.0001
        self.neg_iou_thresh = 0.3

        # create array of sample indices. maps
        # position (index in sample) -> true index

        # TODO: Patrick: i changed the script to work with a sample
        # file--a file where the number on line i is the true index j
        # of the i-th element of the sample. you have to generate one if you're
        # working with a sample, or change the file to use all the data.
        indices = []
        with open("sample_idx_map.txt", 'r') as samp_map:
            for line in samp_map:
                indices.append(int(line))

        self.sample_map = indices
        global SAMPLE_MAP 
        SAMPLE_MAP = indices

        dim = net.params['cls_score'][0].shape[0]
        scale = self._get_feature_scale()
        print('Feature dim: {}'.format(dim))
        print('Feature scale: {:.3f}'.format(scale))
        self.trainers = [SVMClassTrainer(cls, dim, feature_scale=scale)
                         for cls in imdb.classes]

    def _get_feature_scale(self, num_images=100):
        TARGET_NORM = 20.0 # Magic value from traditional R-CNN
        _t = Timer()
        roidb = self.imdb.roidb
        total_norm = 0.0
        count = 0.0

        # choose random indices, 
        inds = npr.choice(self.sample_map, size=num_images,
                          replace=False)
        for i_, i in enumerate(inds):
            #im = cv2.imread(self.imdb.image_path_at(i))
            #if roidb[i]['flipped']:
            #im = im[:, ::-1, :]
            _t.tic()
            scores, boxes, feat = im_detect2(self.net, i, len(roidb[i]['boxes']), False)
            _t.toc()

            total_norm += np.sqrt((feat ** 2).sum(axis=1)).sum()
            count += feat.shape[0]
            print('{}/{}: avg feature norm: {:.3f}'.format(i_ + 1, num_images,
                                                           total_norm / count))

        return TARGET_NORM * 1.0 / (total_norm / count)

    def _get_pos_counts(self):
        counts = np.zeros((len(self.imdb.classes)), dtype=np.int)
        roidb = self.imdb.roidb
        #        for i in xrange(len(roidb)):
        for i in self.sample_map:
            for j in xrange(1, self.imdb.num_classes):
                I = np.where(roidb[i]['gt_classes'] == j)[0]
                counts[j] += len(I)

        for j in xrange(1, self.imdb.num_classes):
            print('class {:s} has {:d} positives'.
                  format(self.imdb.classes[j], counts[j]))

        return counts

    def get_pos_examples(self):
        counts = self._get_pos_counts()
        for i in xrange(len(counts)):
            self.trainers[i].alloc_pos(counts[i])

        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)
        # num_images = 100
        for idx, i in enumerate(self.sample_map):
            gt_inds = np.where(roidb[i]['gt_classes'] > 0)[0]
            gt_boxes = roidb[i]['boxes'][gt_inds]
            _t.tic()
            scores, boxes, feat = im_detect2(self.net, i, gt_inds, False)
            _t.toc()

            for j in xrange(1, self.imdb.num_classes):
                cls_inds = np.where(roidb[i]['gt_classes'][gt_inds] == j)[0]
                if len(cls_inds) > 0:
                    cls_feat = feat[cls_inds, :]
                    self.trainers[j].append_pos(cls_feat)

            print 'get_pos_examples: {:d}/{:d} ({}/{}) {:.3f}s' \
                  .format(i + 1, len(roidb), idx, len(self.sample_map), _t.average_time)

    def initialize_net(self):
        # Start all SVM parameters at zero
        self.net.params['cls_score'][0][...] = 0
        self.net.params['cls_score'][1][...] = 0

        # Initialize SVMs in a smart way. Not doing this because its such
        # a good initialization that we might not learn something close to
        # the SVM solution.
#        # subtract background weights and biases for the foreground classes
#        w_bg = self.net.params['cls_score'][0].data[0, :]
#        b_bg = self.net.params['cls_score'][1].data[0]
#        self.net.params['cls_score'][0].data[1:, :] -= w_bg
#        self.net.params['cls_score'][1].data[1:] -= b_bg
#        # set the background weights and biases to 0 (where they shall remain)
#        self.net.params['cls_score'][0].data[0, :] = 0
#        self.net.params['cls_score'][1].data[0] = 0

    def update_net(self, cls_ind, w, b):
        self.net.params['cls_score'][0][:, cls_ind] = w
        self.net.params['cls_score'][1][0, cls_ind] = b

    def train_with_hard_negatives(self):
        _t = Timer()
        roidb = self.imdb.roidb
        num_images = len(roidb)
        # num_images = 100
        for idx, i in enumerate(self.sample_map):
#            im = cv2.imread(self.imdb.image_path_at(i))
#            if roidb[i]['flipped']:
#                im = im[:, ::-1, :]
            _t.tic()
            scores, boxes, feat = im_detect2(self.net, i, len(roidb[i]['boxes']))
            _t.toc()

            print scores.shape

            # start at 1--don't worry about background class
            for j in xrange(1, self.imdb.num_classes):
                hard_inds = \
                    np.where((scores[:, j] > self.hard_thresh) &
                             (roidb[i]['gt_overlaps'][:, j].toarray().ravel()[0:scores.shape[0]] <
                              self.neg_iou_thresh))[0]
                if len(hard_inds) > 0:
                    hard_feat = feat[hard_inds, :].copy()
                    new_w_b = \
                        self.trainers[j].append_neg_and_retrain(feat=hard_feat)
                    if new_w_b is not None:
                        self.update_net(j, new_w_b[0], new_w_b[1])
                        np.savetxt("svmweights.txt", net.params['cls_score'][0])
                        np.savetxt("svmbias.txt", net.params['cls_score'][1])

            print(('train_with_hard_negatives: '
                   '{:d}/{:d} ({}) {:.3f}s').format(idx + 1, len(self.sample_map), i,
                                               _t.average_time))

    def train(self):
        # Initialize SVMs using
        #   a. w_i = fc8_w_i - fc8_w_0
        #   b. b_i = fc8_b_i - fc8_b_0
        #   c. Install SVMs into net
        self.initialize_net()

        # Pass over roidb to count num positives for each class
        #   a. Pre-allocate arrays for positive feature vectors
        # Pass over roidb, computing features for positives only
        self.get_pos_examples()

        # Pass over roidb
        #   a. Compute cls_score with forward pass
        #   b. For each class
        #       i. Select hard negatives
        #       ii. Add them to cache
        #   c. For each class
        #       i. If SVM retrain criteria met, update SVM
        #       ii. Install new SVM into net
        self.train_with_hard_negatives()

        # One final SVM retraining for each class
        # Install SVMs into net
        for j in xrange(1, self.imdb.num_classes):
            new_w_b = self.trainers[j].append_neg_and_retrain(force=True)
            self.update_net(j, new_w_b[0], new_w_b[1])

class SVMClassTrainer(object):
    """Manages post-hoc SVM training for a single object class."""

    def __init__(self, cls, dim, feature_scale=1.0,
                 C=0.001, B=10.0, pos_weight=2.0):
        self.pos = np.zeros((0, dim), dtype=np.float32)
        self.neg = np.zeros((0, dim), dtype=np.float32)
        self.B = B
        self.C = C
        self.cls = cls
        self.pos_weight = pos_weight
        self.dim = dim
        self.feature_scale = feature_scale
        self.svm = svm.LinearSVC(C=C, class_weight='balanced',#class_weight={1: 2, -1: 1},
                                 intercept_scaling=B, verbose=1,
                                 penalty='l2', loss='l1',
                                 random_state=cfg.RNG_SEED, dual=True)
        self.pos_cur = 0
        self.num_neg_added = 0
        self.retrain_limit = 2000
        self.evict_thresh = -1.1
        self.loss_history = []

    def alloc_pos(self, count):
        self.pos_cur = 0
        self.pos = np.zeros((count, self.dim), dtype=np.float32)

    def append_pos(self, feat):
        num = feat.shape[0]
        self.pos[self.pos_cur:self.pos_cur + num, :] = feat
        self.pos_cur += num

    def train(self):
        print('>>> Updating {} detector <<<'.format(self.cls))
        num_pos = self.pos.shape[0]
        num_neg = self.neg.shape[0]
        print('Cache holds {} pos examples and {} neg examples'.
              format(num_pos, num_neg))
        X = np.vstack((self.pos, self.neg)) * self.feature_scale
        y = np.hstack((np.ones(num_pos),
                       -np.ones(num_neg)))
        self.svm.fit(X, y)
        w = self.svm.coef_
        b = self.svm.intercept_[0]
        scores = self.svm.decision_function(X)
        pos_scores = scores[:num_pos]
        neg_scores = scores[num_pos:]

        num_neg_wrong = len(np.where((neg_scores > 0))[0])
        num_pos_wrong = len(np.where((pos_scores < 0))[0])
        print("pos wrong: {}/{}; neg wrong: {}/{}".format(num_pos_wrong, num_pos, num_neg_wrong, num_neg))

        pos_loss = (self.C * self.pos_weight *
                    np.maximum(0, 1 - pos_scores).sum())
        neg_loss = self.C * np.maximum(0, 1 + neg_scores).sum()
        reg_loss = 0.5 * np.dot(w.ravel(), w.ravel()) + 0.5 * b ** 2
        tot_loss = pos_loss + neg_loss + reg_loss
        self.loss_history.append((tot_loss, pos_loss, neg_loss, reg_loss))

        for i, losses in enumerate(self.loss_history):
            print(('    {:d}: obj val: {:.3f} = {:.3f} '
                   '(pos) + {:.3f} (neg) + {:.3f} (reg)').format(i, *losses))

        # Sanity check
        scores_ret = (
                X * 1.0 / self.feature_scale).dot(w.T * self.feature_scale) + b
        assert np.allclose(scores, scores_ret[:, 0], atol=1e-5), \
                "Scores from returned model don't match decision function"

        return ((w * self.feature_scale, b), pos_scores, neg_scores)

    def append_neg_and_retrain(self, feat=None, force=False):
        if feat is not None:
            num = feat.shape[0]
            self.neg = np.vstack((self.neg, feat))
            self.num_neg_added += num
        if self.num_neg_added > self.retrain_limit or force:
            self.num_neg_added = 0
            new_w_b, pos_scores, neg_scores = self.train()
            # scores = np.dot(self.neg, new_w_b[0].T) + new_w_b[1]
            # easy_inds = np.where(neg_scores < self.evict_thresh)[0]
            not_easy_inds = np.where(neg_scores >= self.evict_thresh)[0]
            if len(not_easy_inds) > 0:
                self.neg = self.neg[not_easy_inds, :]
                # self.neg = np.delete(self.neg, easy_inds)
            print('    Pruning easy negatives')
            print('    Cache holds {} pos examples and {} neg examples'.
                  format(self.pos.shape[0], self.neg.shape[0]))
            print('    {} pos support vectors'.format((pos_scores <= 1).sum()))
            print('    {} neg support vectors'.format((neg_scores >= -1).sum()))
            return new_w_b
        else:
            return None

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train SVMs (old skool)')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Must turn this off to prevent issues when digging into the net blobs to
    # pull out features (tricky!)
    cfg.DEDUP_BOXES = 0

    # Must turn this on because we use the test im_detect() method to harvest
    # hard negatives
    cfg.TEST.SVM = True

    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    # fix the random seed for reproducibility
    np.random.seed(cfg.RNG_SEED)

    """
    # set up caffe
    caffe.set_mode_gpu()
    if args.gpu_id is not None:
        caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]
    out = os.path.splitext(os.path.basename(args.caffemodel))[0] + '_svm'
    out_dir = os.path.dirname(args.caffemodel)
    """



    imdb = get_imdb(args.imdb_name)

    """    fh = open('smallsamp.txt', 'w')
    fh1 = open('smallsamp.rois.txt', 'w')

    num_samp = 100
    indices = [random.choice(range(0,imdb.num_images)) for x in range(0, num_samp)]

    # write out cntk format training file for the images.
    for i, idx in enumerate(indices):
        path = imdb.image_path_at(idx)
        fh.write(str(i) + "\t" + path + "\t0\n")
        im = cv2.imread(path)
        h,w = im.shape[0:2]

        mindim = np.min([w,h])
        maxdim = np.max([w,h])

        scale_factor = 224. / float(mindim)
        scaled_max = maxdim * scale_factor
 
        crop_x, crop_y = False, False

        if maxdim == w:
            crop_x = True
        else:
            crop_y = True

        crop_offset = round((maxdim*scale_factor - 224) / 2.)

        if crop_offset < 0:
            crop_offset = 0

        boxes = ""
        box_counter = 0
        for box in imdb.roidb[idx]['boxes']:

            # only keep 2000 boxes per image.
            # todo: make sure you keep ground truth.
            if box_counter == 2000:
                break

            # fix output bounding boxes to account for the scale & center crop.
            x, y, xmax, ymax = np.asarray(box) * scale_factor

            # put the box in crop coordinates
            # todo: xmax < crop_offset; x > crop_offset + 224
            if crop_x:
                if x < crop_offset:
                    x = 0
                else:
                    x = x - crop_offset
                if x > 224:
                    x = 224

                if xmax > crop_offset + 224:
                    xmax = 224
                elif xmax > crop_offset:
                    xmax = xmax - crop_offset
                else:
                    #print "got box with xmax < crop offset."
                    #print box
                    # print [w,h]
                    xmax = 0

            elif crop_y:
                if y < crop_offset:
                    y = 0
                else:
                    y = y - crop_offset
                if y > 224:
                    y = 224
                if ymax > crop_offset + 224:
                    ymax = 224
                elif ymax > crop_offset:
                    ymax = ymax - crop_offset
                else:
                    ymax = 0

            xrel = float(x) / 224.
            yrel = float(y) / 224.
            wrel = float(xmax - x) / 224.
            hrel = float(ymax - y) / 224.

            assert xrel <= 1.0, "something wrong with xrel"
            assert yrel <= 1.0, "something wrong with yrel"
            assert wrel >= 0.0, "wrel can't be < 0: xmax {}, x {}".format(xmax, x)
            assert hrel >= 0.0, "hrel can't be < 0"

            boxes += " {} {} {} {}".format(xrel, yrel, wrel, hrel)
            box_counter+=1

        # if we have less than 2000 rois per image, fill in the rest.
        while box_counter < 2000:
            boxes += " 0 0 0 0"
            box_counter+=1

        fh1.write(str(i) + " |rois" + boxes + "\n")

    fh1.close()
    fh.close()
    """

    net = DummyNet(4096, imdb)

    print 'Loaded dataset `{:s}` for training'.format(imdb.name)

    global LINE_OFFSETS
    with open("offsets.txt") as file:
        for line in file:
            LINE_OFFSETS.append(int(line))

    # (hjl): don't use flipped for now...

    # enhance roidb to contain flipped examples
#    if cfg.TRAIN.USE_FLIPPED:
#        print 'Appending horizontally-flipped training examples...'
#        imdb.append_flipped_images()
#        print 'done'


    SVMTrainer(net, imdb).train()

    np.savetxt("svmweights.txt", net.params['cls_score'][0])
    np.savetxt("svmbias.txt", net.params['cls_score'][1])
#    print 'Wrote svm model to: {:s}'.format(filename)
