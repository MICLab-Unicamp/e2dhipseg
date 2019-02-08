# -*- encoding: utf-8 -*-
# Module iablob

import numpy as np
import ia870

def iablob(fr, measurement, option="image"):

    measurement = measurement.upper()
    option      = option.upper()
    if len(fr.shape) == 1: fr = fr[newaxis,:]
    n = fr.max()
    if   option      == 'DATA':        y = []
    elif measurement == 'CENTROID':    y = np.zeros(fr.shape,bool)
    elif measurement == 'BOUNDINGBOX': y = np.zeros(fr.shape,bool)
    elif measurement == 'PERIMETER':   y = np.zeros(fr.shape,np.int32)
    elif measurement == 'AREA':        y = np.zeros(fr.shape,np.int32)
    else:                              y = np.zeros(fr.shape)

    if measurement == 'AREA':
        area = np.bincount(fr.ravel())
        if option == 'DATA':
            y = area[1::]
        else:
            for i in range(1,n+1):
               y[fr==i] = area[i]

    elif measurement == 'CENTROID':
        for i in range(1,n+1):
            aux  = (fr==i)
            xind,yind = np.nonzero(aux)
            area = len(xind)
            centroid = [xind.sum()/area,yind.sum()/area]
            if option == 'DATA': y.append([centroid[1],centroid[0]])
            else               : y[centroid[0],centroid[1]] = 1

    elif measurement == 'BOUNDINGBOX':
        for i in range(1,n+1):
            aux = (fr==i)
            col, = np.nonzero(aux.any(0))
            row, = np.nonzero(aux.any(1))
            if option == 'DATA': y.append([col[0],row[0],col[-1],row[-1]])
            else:
                y[row[0]:row[-1],col[0] ] = 1
                y[row[0]:row[-1],col[-1]] = 1
                y[row[0], col[0]:col[-1]] = 1
                y[row[-1],col[0]:col[-1]] = 1

    elif measurement == 'PERIMETER':
        Bc = ia870.iasecross()
        for i in range(1,n+1):
           aux = fr == i
           grad = aux - ia870.iaero(aux,Bc)
           if option == 'DATA': y.append(grad.sum())
           else:
               y[aux] = grad.sum()

    elif measurement == 'CIRCULARITY':
        Bc = ia870.iasecross()
        area = np.bincount(fr.ravel())
        perim = []
        for i in range(1,n+1):
           aux = fr == i
           grad = aux - ia870.iaero(aux,Bc)
           perim.append(grad.sum())
           if option != 'DATA':
               y[aux] = 4*np.pi*area[i]/(perim[i-1]**2)
        if option == 'DATA':
            perim = np.array(perim)
            y = 4*np.pi*area[1::]/(perim**2)

    elif measurement == 'ASPECTRATIO':
        for i in range(1,n+1):
            aux = (fr==i)
            col, = np.nonzero(aux.any(0))
            row, = np.nonzero(aux.any(1))
            if option == 'DATA': y.append(1.*(min(col[-1]-col[0],row[-1]-row[0])+1)/(max(col[-1]-col[0],row[-1]-row[0])+1))
            else:
                y[aux] = 1.*(min(col[-1]-col[0],row[-1]-row[0])+1)/(max(col[-1]-col[0],row[-1]-row[0])+1)

    elif measurement == 'COMPACTNESS':
        for i in range(1,n+1):
            aux  = (fr==i)
            xind,yind = np.nonzero(aux)
            area = len(xind)
            centroid = [xind.sum()/area,yind.sum()/area]
            m20 = ((1.*xind-centroid[0])**2).sum()/area
            m02 = ((1.*yind-centroid[1])**2).sum()/area
            compactness = area/(2*np.pi*(m20+m02))
            if option == 'DATA': y.append(compactness)
            else               : y[aux] = compactness

    elif measurement == 'ECCENTRICITY':
        for i in range(1,n+1):
            aux  = (fr==i)
            xind,yind = np.nonzero(aux)
            area = len(xind)
            centroid = [xind.sum()/area,yind.sum()/area]
            m11 = ((1.*xind-centroid[0])*(yind-centroid[1])).sum()/area
            m20 = ((1.*xind-centroid[0])**2).sum()/area
            m02 = ((1.*yind-centroid[1])**2).sum()/area
            eccentricity = sqrt((m20-m02)**2+4*m11**2)/(m20+m02)
            if option == 'DATA': y.append(eccentricity)
            else               : y[aux] = eccentricity

    else:
        print("Measurement option should be 'AREA','CENTROID', 'BOUNDINGBOX', 'PERIMETER', ")
        print("'ASPECTRATIO', 'CIRCULARITY', 'COMPACTNESS' or 'ECCENTRICITY'.")

    return np.array(y)

