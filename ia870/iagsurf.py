# -*- encoding: utf-8 -*-
# Module iagsurf

def iagsurf(f, vx=1, vy=0, vz=1):

    import numpy as np
    import ia636
    zd = ia636.iapad(f).astype(float)
    gv = zd[1:-1,2:] - zd[1:-1,1:-1]
    gv /= gv.max()
    #adshow(ia636.ianormalize(gv))
    gh = zd[2:,1:-1] - zd[1:-1,1:-1]
    gh /= gh.max()
    #adshow(ia636.ianormalize(gh))
    gz = 1.0/(gv**2 + gh**2 + 1)
    #adshow(ia636.ianormalize(gz))
    gv *= gz
    gh *= gz
    #adshow(ia636.ianormalize(gv))
    #adshow(ia636.ianormalize(gh))
    v = np.sqrt(vx*vx + vy*vy + vz*vz)
    vx = vx/v; vy = vy/v; vz = vz/v;
    gv *= vx
    gh *= vy
    gz *= vz
    gz += gh + gv
    return ia636.ianormalize(gz)

