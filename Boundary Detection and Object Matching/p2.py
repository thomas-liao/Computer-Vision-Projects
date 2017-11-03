import numpy as np
class EquivalenceTable:

    def __init__(self, n):
        self._id = list(range(n))
        for i in range(n):
            self._id[i] = i

    # find root with path compression
    def find_root(self, i):
        j = i
        while j != self._id[j]:
            self._id[j] = self._id[self._id[j]]
            j = self._id[j]
        return j

    def union(self, p, q):
        i = self.find_root(p)
        j = self.find_root(q)
        if i == j:
            return
        if i < j:
            self._id[j] = i
        else:
            self._id[i] = j



def p2(binary_in): # return labels_out
    height = len(binary_in)
    width = len(binary_in[0])
    n = height * width

    # must use np.unit 16 instead of np.unit 8
    ret_img = np.zeros((height, width), dtype = np.uint64) # bug here, must use np.unit64 instead of np.unit 8, otherwise causing bug for not being able to distinguishing all the labels
    lb = 0 # label
    eqtb = EquivalenceTable(99999)

    # 1st path
    for i in range(height):
        for j in range(width):
            # find seed 1
            if binary_in[i][j] == 0:
                continue
            #get information from up and left

            up = ret_img[i-1][j]
            left = ret_img[i][j-1]

            # case 1, left = up = 0, assign new label
            if left == 0 and up == 0:
                lb += 1
                ret_img[i][j] = lb
            # case 2, 3: one and only one of left and up is 0
            if left != 0  and up == 0:
                ret_img[i][j] = left
            if left == 0 and up != 0:
                ret_img[i][j] = up
            # case 4: both not 0
            if left != 0 and up != 0:
                # case 4.1: equal
                if left == up:
                    ret_img[i][j] = left
                # case 4.2: not equal, label conflict handling
                else:
                    eqtb.union(left, up)
                    ret_img[i][j] = left

    #second path
    for i in range(height):
        for j in range(width):
            # if background, continue
            if (binary_in[i][j] == 0):
                continue
            ret_img[i][j] = eqtb._id[eqtb.find_root(ret_img[i][j])]

##########################################################################################################################################

    # THIS PART is OPTIONAL.. can be deleted or commented out!! (especially if it is too slow..)!
    #
    # # by so far we can already return the image, but the only problem is, the labels are sparsely distributed,
    # # which is not very good for display purpose
    # # So here I'm writing loops to address this issue, just for fun, not necessary
    transform = dict()
    new_key = 1
    for i in range(height):
        for j in range(width):
            k = ret_img[i][j]
            if k in transform or k == 0:
                continue
            else:
                transform[k] = new_key
                new_key += 1
    for i in range(height):
        for j in range(width):
            k = ret_img[i][j]
            if k == 0:
                continue
            ret_img[i][j] = transform[k]
###########################################################################################################################################
    return ret_img
