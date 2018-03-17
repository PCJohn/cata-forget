import numpy as np

def permute(ds,vds):
    rp = range(28)
    cp = range(28)
    np.random.shuffle(rp)
    np.random.shuffle(cp)
    tds,tvds = [],[]
    for x,y in ds:
        tds.append((x[:,rp,:][:,:,cp],y))
    for vx,vy in vds:
        tvds.append((vx[:,rp,:][:,:,cp],vy))
    return (tds,tvds)
