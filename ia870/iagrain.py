# -*- encoding: utf-8 -*-
# Module iagrain

import numpy as np

def iagrain(fr, f, measurement, option="image"):

    measurement = measurement.upper()
    option      = option.upper()
    if fr.ndim == 1: fr = fr[newaxis,:]
    n = fr.max()
    if option == 'DATA': y = np.empty((n,),np.float)
    else               : y = np.zeros(fr.shape)
    if measurement == 'MAX':
        for i in range(1,n+1):
            val = f[fr==i].max()
            if option == 'DATA': y[i-1] = val
            else               : y[fr==i] = val
    elif measurement == 'MIN':
        for i in range(1,n+1):
            val = f[fr==i].min()
            if option == 'DATA': y[i-1] = val
            else               : y[fr==i] = val
    elif measurement == 'SUM':
        for i in range(1,n+1):
            val = f[fr==i].sum()
            if option == 'DATA': y[i-1] = val
            else               : y[fr==i] = val
    elif measurement == 'MEAN':
        for i in range(1,n+1):
            val = f[fr==i].mean()
            if option == 'DATA': y[i-1] = val
            else               : y[fr==i] = val
    elif measurement == 'STD':
        for i in range(1,n+1):
            v = f[fr==i].std()
            if len(v) < 2: val = 0
            else         : val = v.std()
            if option == 'DATA': y[i-1] = val
            else               : y[fr==i] = val
    elif measurement == 'STD1':
        print("STD1 is not implemented")
    else:
        print("Measurement should be 'MAX', 'MIN', 'MEAN', 'SUM', 'STD', 'STD1'.")
    return y

