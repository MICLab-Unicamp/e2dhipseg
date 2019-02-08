# -*- encoding: utf-8 -*-
# Module iadrawv

from numpy import *

def iadrawv(f, data, value, GEOM):

    GEOM  = GEOM.upper()
    data  = array(data)
    y     = array(f)
    value = array(value)
    lin, col = data[1,:], data[0,:]
    i = lin*f.shape[1] + col
    i = i.astype(int32)
    if len(f.shape) == 1: f = f[newaxis,:]
    if value.shape == (): value = value + zeros(lin.shape)
    if len(lin) != len(value):
        print('Number of points must match n. of colors.')
        return None
    if GEOM == 'POINT':
        ravel(y)[i] = value
    elif GEOM == 'LINE':
        for k in range(len(value)-1):
            delta = 1.*(lin[k+1]-lin[k])/(1e-10 + col[k+1]-col[k])
            if abs(delta) <= 1:
                if col[k] < col[k+1]: x_ = arange(col[k],col[k+1]+1)
                else                : x_ = arange(col[k+1],col[k]+1)
                y_ = floor(delta*(x_-col[k]) + lin[k] + 0.5)
            else:
                if lin[k] < lin[k+1]: y_ = arange(lin[k],lin[k+1]+1)
                else                : y_ = arange(lin[k+1],lin[k]+1)
                x_ = floor((y_-lin[k])/delta + col[k] + 0.5)
            i_ = y_*f.shape[1] + x_; i_ = i_.astype(int32)
            ravel(y)[i_]=value[k]
    elif GEOM == 'RECT':
        for k in range(data.shape[1]):
            d = data[:,k]
            x0,y0,x1,y1 = d[1],d[0],d[3],d[2]
            y[x0:x1,y0]   = value[k]
            y[x0:x1,y1]   = value[k]
            y[x0,y0:y1]   = value[k]
            y[x1,y0:y1+1] = value[k]
    elif GEOM == 'FRECT':
        for k in range(data.shape[1]):
            d = data[:,k]
            x0,y0,x1,y1 = d[1],d[0],d[3],d[2]
            y[x0:x1+1,y0:y1+1] = value[k]
    else:
        print("GEOM should be 'POINT', 'LINE', 'RECT', or 'FRECT'.")
    return y

