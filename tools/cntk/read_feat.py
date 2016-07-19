"""
Script that reads in an offset file (generated with make_offsets)
and a big feature file (the one input to make_offsets) and creates
one feature file per image.

python read_feat.py <offset_file> <feature_file>

"""

import cPickle as pickle
from scipy.sparse import csr_matrix
from scipy import io
from array import array
import numpy as np
import sys

sample_map = []

OUTDIR = '/mnt/d/'
offset_file = sys.argv[1]
feature_file = sys.argv[2]

offsets = []
with open(offset_file) as off:
    for line in off:
        offsets.append(int(line))

INF = open(feature_file)


for idx in range(0, len(offsets)):
#    with open("/mnt/d/sampfeat/{}.arr".format(idx)) as inf:
    if True:
        """
        float_arr = array('d')
        float_arr.fromstring(inf.read())
        imfeat = np.array(float_arr.tolist(), dtype='float').reshape((2000, 4096))
        sparse = csr_matrix(imfeat)

        outp = open('/mnt/d/samppkl/{}.pkl'.format(idx), 'wb')
        pickle.dump(sparse, outp)
        outp.close()
        
        io.mmwrite('/mnt/d/sampmm/{}.mm'.format(idx), sparse)
        """

        feat = np.zeros((2000, 4096))
        first_line = 2000*idx
        offset = offsets[first_line]
        size = offsets[first_line + 1999] - offset

        INF.seek(offset)
        lines = INF.readlines(size)
        for i, line in enumerate(lines):
            feat[i, :] = np.fromstring(line, dtype=float, sep=' ')

        outa = open('{}{}.arr'.format(OUTDIR,idx), 'wb')
        float_array = array('d', feat.flatten())
        float_array.tofile(outa)
        outa.close()
