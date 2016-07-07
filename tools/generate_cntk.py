import _init_paths
from datasets.factory import get_imdb
import cv2
import numpy as np
import sys
from PIL import Image, ImageDraw

def resize_and_crop(img_path, modified_path, size, crop_type='top'):
    """
    Resize and crop an image to fit the specified size.
    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'midle' or
            'bottom/rigth' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    #The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], size[0] * img.size[1] / img.size[0]),
                Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, (img.size[1] - size[1]) / 2, img.size[0], (img.size[1] + size[1]) / 2)
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((size[1] * img.size[0] / img.size[1], size[1]),
                Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = ((img.size[0] - size[0]) / 2, 0, (img.size[0] + size[0]) / 2, img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else :
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else :
        img = img.resize((size[0], size[1]),
                Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img.save(modified_path)


imdb = get_imdb('voc_2012_trainval')

fh = open('trainval2012.txt', 'w')
fh1 = open('trainval2012.rois.txt', 'w')

#num_samp = 100
#indices = [random.choice(range(0,imdb.num_images)) for x in range(0, num_samp)]

# write out cntk format training file for the images.
for i, idx in enumerate(range(0, imdb.num_images)):

    path = imdb.image_path_at(idx)

    #resize_and_crop(path, 'cropped.png', (224, 224), crop_type='middle')

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
        # todo: make sure you keep good boxes.
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
                continue
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
                continue
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
