import numpy as np
from array import array
from timeit import default_timer as timer


offsets = []
with open('test2007-offsets.txt') as off:
    for line in off:
        offsets.append(int(line))

INF = open('/mnt/d/test2007.h2.y')

start = timer()
feat1 = np.zeros((4000,4096))

for i in range(0, 4000):
    feat1[i, :] = np.fromstring(INF.readline(), dtype=float, sep=' ')
end = timer()
print(end - start)


feat1 = feat1[0:2000, :]

start = timer()
float_arr = array('d')
float_arr.fromfile(open('/mnt/d/testfeat/1.arr'), 2000*4096)
feat2 = np.array(float_arr.tolist(), dtype='float').reshape((2000, 4096))
end = timer()
print(end - start)

print len(np.where(feat1 != feat2)[0])
