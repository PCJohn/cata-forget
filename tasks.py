#import cv2
import numpy as np

def permute(ds,vds):
    rp = range(28)
    cp = range(28)
    #a = rp[:14]
    #b = rp[14:]
    #np.random.shuffle(a)
    #rp = a+b
    np.random.shuffle(rp)
    np.random.shuffle(cp)
    tds,tvds = [],[]
    for x,y in ds:
        #cv2.imshow('help',x[:,rp,:].transpose(2,1,0))
        #cv2.imshow('help2',x[:,:,:].transpose(2,1,0))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        tds.append((x[:,rp,:][:,:,cp],y))
    for vx,vy in vds:
        tvds.append((vx[:,rp,:][:,:,cp],vy))
    return (tds,tvds)
