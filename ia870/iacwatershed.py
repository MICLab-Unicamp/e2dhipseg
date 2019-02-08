# -*- encoding: utf-8 -*-
# Module iacwatershed

from ia870 import iasecross, iaisbinary, ialabel, iasubm, iaero, iabinary

def iacwatershed(f, g, Bc=iasecross(), option='LINES'):
    from ipdp import se2offset

    if iaisbinary(g):
        g = ialabel(g, Bc)
    offset = se2offset(Bc)
    w = ift_m(f, offset, g)
    if option == 'LINES':
        w = iasubm(w, iaero(w))
        w = iabinary(w)
    return w

def ift_m(im, offsets, M):

    from ipdp import wsImage

    # initialise variables
    ws = wsImage(im)
    N, im, lab, D = ws.begin(offsets)

    wsM = wsImage(M)
    NM, imM, labM, DM = wsM.begin(offsets)

    # make the set
    Mset = dict()
    for p in D:
        if imM[p] > 0:
            if Mset.has_key(imM[p]):
                Mset[imM[p]].append(p)
            else:
                Mset[imM[p]] = [p]

    ift_k(ws, im, Mset.values(), N, D, lab)

    return ws.end()

# constants
MASK = -2

def ift_k(ws, im, M, N, D, lab):

    import numpy as np
    from ipdp import wsHeapQueue

    # create the working images
    done = ws.makeWorkCopy(False)
    c1 = ws.makeWorkCopy(np.inf)
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

