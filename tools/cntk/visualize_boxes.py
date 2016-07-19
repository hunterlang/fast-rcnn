import _init_paths
from datasets.factory import get_imdb
import cv2
import numpy as np
import sys
from PIL import Image, ImageDraw

imdb = get_imdb('voc_2007_test')

image_index = 581
feat_file = open('/mnt/d/planeGPU-big.OutputNodes.z')

# load scores
scores = np.zeros((2000, 1000))
for i, line in enumerate(feat_file):
    scores[i, :] = np.fromstring(line, dtype=float, sep=' ')

labels = np.argmax(scores, axis=1)
print scores[np.where(labels == 895)]
# get labels
scores = scores[:, 895]


maxval = np.max(scores)
threshold = maxval*0.9

print labels[np.where(scores > threshold)]

rois = imdb.roidb[image_index]

# imagenet dog classes
doglabels = range(151, 269)
doglabels = dict(zip(doglabels, [0 for x in range(0, len(doglabels))]))

planelabel = 895

im = Image.open(imdb.image_path_at(image_index))
w = im.size[0]
h = im.size[1]

scale_factor = h/224.
crop = (w - 224)/2 / scale_factor

draw = ImageDraw.Draw(im)
draw.rectangle([(crop, 0), (w - (224./scale_factor - crop), h)], outline='black')

for i, box in enumerate(rois['boxes']):
    if i >= 2000:
        continue

    bbox = [(box[0], box[1]), (box[2], box[3])]
    area = int(box[2] - box[0]) * int(box[3] - box[1])

    if area <= 1000:
        continue

    draw = ImageDraw.Draw(im)

    boxcol = 'green' if (scores[i] > threshold and (labels[i] == planelabel or labels[i] == 404)) else 'red'

    if boxcol == 'green':
        draw.rectangle(bbox, outline=boxcol)
    del draw

im.save('planeboxes_green.png')
