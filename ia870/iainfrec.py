# -*- encoding: utf-8 -*-
# Module iainfrec

import numpy as np
from ia870.iasecross import iasecross
import ia870 as MT

def iainfrec_eq(f, g, bc=iasecross()):
    n = f.size
    y = MT.iacdil(f,g,bc,n)
    return y

def iainfrec(m, f, Bc=iasecross()):
    h,w = Bc.shape
    hc,wc = h/2,w/2
    B = Bc.copy()
    off = np.transpose(B.nonzero()) - np.array([hc,wc])
    i = off[:,0] * w + off[:,1]
    Nids = MT.iaNlut(f.shape,off)
    x,y = np.where(Nids==f.size)
    Nids[x,y] = x
    Nids_pos = Nids[:,i<0] #de acordo com a convenção em Vincent93
    Nids_neg = Nids[:,i>0] #de acordo com a convenção em Vincent93

    I = f.flatten()
    J = m.flatten()
    D = np.nonzero(J)[0]
    V = np.zeros(f.size,np.bool) #para controle de inserção na fila
    fila =[]

    for p in D:
        Jq = J[p]
        for q in Nids_pos[p]:
            Jq = max(Jq,J[q])
            if (J[q] < J[p]) and (J[q] < I[q]) and ~V[p]:
                fila.append(p)
                V[p]=True
            J[p] = min(Jq,I[p])

    for p in D[::-1]:
        Jq = J[p]
        for q in Nids_neg[p]:
            Jq = max(Jq,J[q])
            if (J[q] < J[p]) and (J[q] < I[q]) and ~V[p]:
                fila.append(p)
                V[p]=True
            J[p] = min(Jq,I[p])

    while fila:
        p = fila.pop(0)
        for q in Nids[p]:
            if J[q]<J[p] and I[q]!=J[q]:
                J[q] = min(J[p],I[q])
                fila.append(q)

    return J.reshape(f.shape)

