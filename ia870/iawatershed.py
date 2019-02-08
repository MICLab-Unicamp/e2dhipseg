# -*- encoding: utf-8 -*-
# Module iawatershed
from ia870.iasecross import iasecross

def iawatershed(f, Bc=iasecross(), option='LINES'):
    from ia870 import iasubm, iaero, iabinary
    from ipdp import se2offset

    offset = se2offset(Bc)
    w = connectedComponents(f, offset)
    if option.upper() == 'LINES':
        w = iasubm(w, iaero(w))
        w = iabinary(w)
    return w

# constants
MASK = -2
PLATEAU = -1

def connectedComponents(im, offsets):

    from ipdp import wsImage
    from ipdp import wsQueue

    # initialise variables
    ws = wsImage(im)
    N, im, lab, D = ws.begin(offsets)

    lab[:] = MASK
    adr = ws.makeWorkCopy(0)

    queue = wsQueue()

    def find(p):
        q = p
        while adr[q] != q:
            q = adr[q]
        u = p
        while adr[u] != q:
            v = adr[u]
            adr[u] = q
            u = v
        return q

    # step 1
    for p in D:

        q = p
        for u in N(p):

            if im[u] < im[q]:
                q = u

        if q != p:
            adr[p] = q
        else:
            adr[p] = PLATEAU

    # step 2
    for p in D:
        if adr[p] != PLATEAU:
            continue

        for q in N(p):
            if adr[q] == PLATEAU or im[q] != im[p]:
                continue

            queue.push(q)

    while not queue.empty():
        p = queue.pop()
        for q in N(p):
            if adr[q] != PLATEAU or im[q] != im[p]:
                continue

            adr[q] = p
            queue.push(q)

    # step 3
    for p in D:
        if adr[p] != PLATEAU:
            continue

        adr[p] = p

        for q in N(p):
            if q > p or im[q] != im[p]:
                continue

            u = find(p)
            v = find(q)
            adr[u] = adr[v] = min(u,v)

    # step 4
    basins = 1
    for p in D:

        r = find(p)
        adr[p] = r
        if lab[r] == MASK:
            lab[r] = basins
            basins += 1
        lab[p] = lab[r]

    return ws.end()

