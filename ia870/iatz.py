# -*- encoding: utf-8 -*-
# Module iatz

from ia870.iasecross import iasecross

# constants
MASK = -2
TIE_ZONE = 0

def iatz(f, Bc=iasecross()):
    from ipdp import se2offset

    offset = se2offset(Bc)
    return tieZone(f, offset)



def tieZone(im, offsets):

    import numpy as np
    from ipdp import wsImage
    from ipdp import findMinima
    from ipdp import wsHeapQueue

    # initialise variables
    ws = wsImage(im)
    N, im, lab, D = ws.begin(offsets)

    # find minima
    M = findMinima(im, N, D)

    # create the working images
    done = ws.makeWorkCopy(False)
    c1 = ws.makeWorkCopy(np.inf)
    c2 = ws.makeWorkCopy(0)
    par = ws.makeWorkCopy(MASK)
    lab[:] = MASK

    queue = wsHeapQueue()

    for m in range(len(M)):
        for p in M[m]:
            c1[p] = im[p]
            lab[p] = m+1
            par[p] = p
            queue.push(p, im[p])

    while not queue.empty():
        p = queue.pop()
        done[p] = True
        for q in N(p):
            if done[q]:
                continue

            c = max(c1[p], im[q])
            if c < c1[q]:
                if c1[q] < np.inf:
                    if queue.contains(q, c1[q]):
                        queue.remove(q, c1[q])
                c1[q] = c
                lab[q] = lab[p]
                par[q] = p
                queue.push(q, c1[q])
                if c == c1[p]:
                    c2[q] = c2[p] + 1
            elif c == c1[q] and lab[q] != lab[p]:
                if c == c1[p]:
                    if c2[q] == c2[p] + 1:
                        lab[q] = TIE_ZONE
                else:
                    lab[q] = TIE_ZONE

    return ws.end()

