import cPickle as pickle
from scipy.sparse import csr_matrix
from scipy import io
from array import array
import numpy as np
import sys

sample_map = []

with open("sample_idx_map.txt") as samp_map:
    for line in samp_map:
        sample_map.append(int(line))


offsets = []
with open('test2007-offsets.txt') as off:
    for line in off:
        offsets.append(int(line))

INF = open('/mnt/d/test2007.h2.y')


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

        outa = open('/mnt/d/testfeat/{}.arr'.format(idx), 'wb')
        float_array = array('d', feat.flatten())
        float_array.tofile(outa)
        outa.close()
