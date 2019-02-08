# -*- encoding: utf-8 -*-
# Module ipdp

from numpy import array, ravel

# constants
# neighbourhood 2D
N4 = array([[-1,0],[0,1],[1,0],[0,-1]])
N8 = array([[-1,0],[0,1],[1,0],[0,-1],[-1,-1],[-1,1],[1,1],[1,-1]])

# neighbourhood 3D
N6 = array([[-1,0,0],[0,1,0],[1,0,0],[0,-1,0],[0,0,1],[0,0,-1]])


# values
inf = 1e400
BORDER = inf

def isBorder(v):
    """ Tests if the value is a border value """
    return v == BORDER

# common classes for the algorithms
class wsImage():
    """ Class for storing the images """

    def __init__(self, array):
        """ Constructor for storing the N-D input image """
        self.input = array
        self.work = None
        self.label = None
        self.output = None

    def begin(self, offsets):
        """ Prepare the image for processing """
        from numpy import zeros, ravel

        if len(self.input.shape) != offsets.shape[1]:
            raise Exception("Image shape does not fit offsets dimensions")

        # initialise the neighbour
        self.neighbour = wsNeighbour(offsets)
        # pad the input image
        self.work = self._pad()
        # store the padded shape
        self.workshape = self.work.shape
        # ravel the padded image (make 1-D)
        self.work = ravel(self.work)
        # make a zeroed copy of it
        self.label = zeros(self.work.shape)
        # initialise the output
        self.output = None
        # initialise the shape of the image
        self.neighbour.setImageShape(self.workshape)
        self.neighbour.setImage(self.work)
        # initialise the domain object
        D = wsDomain(self.work.size)
        D.setImage(self.work)

        # returns the neighbourhood relation, the working image, the label
        # image and the domain of the image
        return self.neighbour.N, self.work, self.label, D


    def _pad(self):
        from numpy import zeros
        """ Pads the N-D image with the BORDER constant as necessary for
        containing all the offsets from the list
        """
        # generate the newshape by iterating through the original and
        # adding the extra space needed at each dimension
        newshape = tuple(map(lambda orig, d: orig + (d-1), self.input.shape, self.neighbour.shape))
        # generate the slicing list
        slicing = map(lambda orig, d: slice((d-1)/2, (d-1)/2 + orig), self.input.shape, self.neighbour.shape)
        # create the padded image
        workimage = zeros(newshape)
        workimage[:] = BORDER
        workimage[slicing] = self.input
        return workimage

    def _crop(self):
        return self.crop(self.label)

    def crop(self, x):
        """ Reshape and crops the N-D label image to the original size (same as self.input) """
        from numpy import reshape
        # generate the slicing list
        slicing = map(lambda orig, d: slice((d-1)/2, (d-1)/2 + orig), self.input.shape, self.neighbour.shape)
        # reshape the label image to the original shape
        temp = reshape(x, self.workshape)
        # crop the temp image
        return temp[slicing]

    def end(self):
        return self._crop().astype('int32')

    def makeWorkCopy(self, default=0):
        """ Make a copy of the work image filled with the value and type of parameter default """
        copied = self.work.copy()
        copied = copied.astype(type(default))
        copied.fill(default)
        return copied

    def getCoordinate(self, p):
        """ Get the coordinates in (x,y) form for index p"""
        return (p / self.workshape[1], p % self.workshape[1])


class wsNeighbour():
    """ Class for neighbourhood processing """

    def __init__(self, offsets):
        """ Constructor for the list of offsets in N-D (neighbours)
            offsets must be a m x N matrix, where m is the number of
            offsets (neighbours) and N is the dimensions of the image"""
        self.offsets = array(offsets)
        self._shape()
        self.s = None

    def _shape(self):
        """ Calculates the shape of the offsets """
        N = self.offsets.shape[1]
        self.shape = []
        for i in range(N):
            dmax = max(self.offsets[:,i])
            dmin = min(self.offsets[:,i])
            if abs(dmax) != abs(dmin):
                raise Exception("Offsets must be symmetrical")
            d = dmax - dmin + 1
            # make the dimension always odd
            if d % 2 == 0:
                d += 1
            self.shape.append(d)
        self.shape = tuple(self.shape)

    def setImageShape(self, imshape):
        """ Set the image shape and calculates the offsets in 1-D """
        self.s = imshape
        if len(self.s) != self.offsets.shape[1]:
            raise Exception("Image shape does not fit offsets dimensions")

        # calculate the offsets in 1-D
        # the process occurs like this:
        # each offset is multiplied by the multiplication of the values of
        # the next components of the shape of the image and summed:
        # example:
        # shape: (3, 10, 15)
        # offset: [1, 1, 2]
        # offset in 1-D: (1 * 10 * 15) + (1 * 15) + (2)
        #
        # of course, the offsets must follow the order of the shape
        # (Nth-D, ..., 3rd-D, 2nd-D, 1st-D), that is usually
        # (time, channel, row, column) or in grayscale images (time, row, column)
        # or simple 2-D images (row, column)

        # LONG VERSION
        # self.roffsets = []
        # for offset in self.offsets:
            # roffset = 0
            # for i in range(len(offset)):
                # n = offset[i]
                # roffset += n * reduce(lambda x,y: x*y, self.s[(i+1):], 1)
            # self.roffsets.append(roffset)

        # SHORT VERSION (using map and reduce - I like this one better)
        self.roffsets = map(
            lambda offset: sum(
                map(lambda n, i:
                        n * reduce(lambda x,y: x*y, self.s[(i+1):], 1),
                    offset, list(range(len(offset)))
                    )
                ),
            self.offsets)

    def setImage(self, im):
        """ Set the working image to query for border values on neighbourhood calculation """
        self.im = im

    def N(self, pixel):
        """ Returns the list of indexes of neighbours of pixel in 1-D """
        if not self.s:
            raise Exception("Set the image shape first!")

        # calculate the coordinates of the neighbours based on the offsets in 1-D
        n = map(lambda c: c + pixel, self.roffsets)
        r = list()

        for i in n:
            if isBorder(self.im[i]):
                continue

            r.append(i)

        return r

    def query(self, c):
        """ Lookup on the roffsets for the index of the offset c """
        for i in range(len(self.roffsets)):
            if self.roffsets[i] == c:
                return i
        return -1

    def addOffset(self, p, index):
        """ Adds the offset of the desired index to the value p """
        if index < 0 or index >= len(self.roffsets):
            return None
        else:
            c = self.roffsets[index]
            return p + c

class wsDomain:
    def __init__(self, length):
        self.inner = range(length)
        self.count = 0

    def next(self):
        if self.count >= len(self.inner):
            self.count = 0
            raise StopIteration

        while isBorder(self.im[self.inner[self.count]]):
            self.count += 1
            if self.count >= len(self.inner):
                self.count = 0
                raise StopIteration

        c = self.count
        self.count = c + 1
        return self.inner[c]

    def __getitem__(self, item):
        return self.inner[item]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.inner)

    def setImage(self, im):
        self.im = im

class wsQueue():
    """ Simple queue class for abstracting list methods """

    def __init__(self):
        self.array = list()

    def push(self, a):
        """ Pushes an element to the end of the queue """
        self.array.append(a)

    def pop(self):
        """ Pops the first element of the queue """
        return self.array.pop(0)

    def empty(self):
        """ Returns whether the queue is empty or not """
        return len(self.array) == 0

    def clear(self):
        """ Clear the list """
        self.array = list()

class wsStack():
    """ Simple stack class for abstracting list methods """

    def __init__(self):
        self.array = list()

    def push(self, a):
        """ Pushes an element to the top of the stack """
        self.array.append(a)

    def pop(self):
        """ Pops the top element of the stack """
        return self.array.pop()

    def empty(self):
        """ Returns whether the stack is empty or not """
        return len(self.array) == 0

    def clear(self):
        """ Clear the list """
        self.array = list()

class wsHeapQueue():
    """ Priority queue class for abstracting list methods with FIFO policy """

    def __init__(self):
        self.queue = dict()

    def push(self, a, c):
        """ Pushes an element to the queue """
        if self.queue.has_key(c):
            self.queue[c].append(a)
        else:
            self.queue[c] = [a]

    def pop(self):
        """ Pops the first element of the queue """
        key = min(self.queue.keys())
        element = self.queue[key].pop(0)
        if len(self.queue[key]) == 0:
            self.queue.pop(key)
        return element

    def empty(self):
        """ Returns whether the queue is empty or not """
        return len(self.queue) == 0

    def clear(self):
        """ Clear the queue """
        self.queue = dict()

    def remove(self, a, c):
        """ Remove the element a at cost c """
        self.queue[c].remove(a)

    def contains(self, a, c):
        """ Verifies if the queue contains element a at cost c """
        for x in self.queue[c]:
            if x == a:
                return True
        return False

class wsRandHeapQueue():
    """ Priority queue class for abstracting list methods without FIFO policy """

    class wsPixel():

        def __init__(self, p, l):
            self.p = p
            self.l = l

        def __lt__(self, other):
            return self.l < other.l

        def __le__(self, other):
            return self.l <= other.l

        def __eq__(self, other):
            return self.l == other.l and self.p == other.p

        def __ne__(self, other):
            return not (self.l == other.l and self.p == other.p)

        def __gt__(self, other):
            return self.l > other.l

        def __ge__(self, other):
            return self.l >= other.l

    def __init__(self):
        self.queue = []

    def push(self, a, c):
        from heapq import heappush
        """ Pushes an element to the queue """
        px = wsRandHeapQueue.wsPixel(a, c)
        heappush(self.queue, px)

    def pop(self):
        from heapq import heappop
        """ Pops the first element of the queue """
        px = heappop(self.queue)
        return px.p

    def empty(self):
        """ Returns whether the queue is empty or not """
        return len(self.queue) == 0

    def clear(self):
        """ Clear the queue """
        self.queue = []

    def remove(self, a, c):
        from heapq import heapify
        """ Remove the element a at cost c """
        self.queue.remove(wsRandHeapQueue.wsPixel(a,c))
        heapify(self.queue)

    def contains(self, a, c):
        """ Verifies if the queue contains element a at cost c """
        for px in self.queue:
            if px.p == a and px.l == c:
                return True
        return False


def findMinima(im, N, D):

    UNVISITED = 0
    PENDING = 1
    VISITED = 2

    work = im.copy()
    work[:] = UNVISITED

    qPending = wsQueue()
    qMinima = wsQueue()

    minima = list()

    count = 1

    for p in D:
        if work[p] != UNVISITED:
            continue

        qPending.push(p)
        work[p] = PENDING

        is_minima = True
        qMinima.clear()

        while not qPending.empty():

            q = qPending.pop()
            qMinima.push(q)
            work[q] = VISITED

            for u in N(q):

                if im[u] == im[q] and work[u] == UNVISITED:
                    work[u] = PENDING
                    qPending.push(u)
                elif im[u] < im[q]:
                    is_minima = False # keep flooding the plateau



        if is_minima:
            count += 1
            seed = list()
            while not qMinima.empty():
                q = qMinima.pop()
                seed.append(q)
            minima.append(seed)

    return minima

def lowerComplete(im, offsets):

    # initialise variables
    ws = wsImage(im)
    N, im, lc, D = ws.begin(offsets)

    FICTITIOUS_PIXEL = -1
    queue = wsQueue()

    lc[:] = BORDER

    for p in D:

        lc[p] = 0
        for q in N(p):
            if im[q] < im[p]:
                queue.push(p)
                lc[p] = -1
                break

    cur_dist = 1

    queue.push(FICTITIOUS_PIXEL)

    while not queue.empty():
        p = queue.pop()
        if p == FICTITIOUS_PIXEL:
            if not queue.empty():
                queue.push(FICTITIOUS_PIXEL)
                cur_dist += 1
        else:
            lc[p] = cur_dist
            for q in N(p):

                if im[q] == im[p] and lc[q] == 0:
                    queue.push(q)
                    lc[q] = -1

    for p in D:

        if lc[p] == 0:
            lc[p] = cur_dist * im[p]
        else:
            lc[p] = cur_dist * im[p] + lc[p] - 1

    return ws.end()

def se2offset(se):
    from numpy import array
    from ia870 import iaseshow

    se = iaseshow(se, 'NORMAL')

    offset = []
    for i in range(se.shape[0]):
        for j in range(se.shape[1]):
            if se[i,j] == True and (i-se.shape[0]/2 != 0 or j-se.shape[1]/2 != 0):
                offset.append([i-se.shape[0]/2,j-se.shape[1]/2])

    return array(offset)

