import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import ewc_fg
import mnist

def permute(ds,vds):
    rp = range(28)
    cp = range(28)
    np.random.shuffle(rp)
    np.random.shuffle(cp)
    tds,tvds = [],[]
    for l,(x,y) in enumerate(ds):
        tds.append((x[:,rp,:][:,:,cp],y))
    for vx,vy in vds:
        tvds.append((vx[:,rp,:][:,:,cp],vy))
    return (tds,tvds)


def run_permute_exp((x,y),(vx,vy),n_tasks,use_ewc=False):
    plots = []
    model = ewc_fg.FullyConn()
    print 'Trask 0: simple MNIST...'
    rem_vec = np.array([1,1,1,1,1,1,1,1,1,1,1])
    ttrc,vtrc = model.train((x,y),rem_vec,val_list=[(vx,vy)],use_ewc=use_ewc)
    #_,vacc = model.test(vx,y=vy)
    print 'Test on task 0',str(vtrc[0][-1])#str(vacc)

    #Permutation tasks
    val_ds = [(vx,vy)]
    #plots.append([(0,vacc)])
    itn = 0
    plots = [[] for _ in range(n_tasks+1)]
    for v in vtrc[0]:
        plots[0].append((itn,v))
        itn += 1
    #accum_val = (list(vx),list(vy))
    for i in range(n_tasks):
        tds,tvds = permute(ds,vds)
        #tds,tvds = ds,vds
        tx,ty = map(np.float32,zip(*tds))
        tvx,tvy = map(np.float32,zip(*tvds))
        val_ds.append((tvx,tvy))
        #val_ds = [(tvx,tvy)]
        #plots.append([])
        print 'Training on task '+str(i+1)+'...'
        if i >= 3:
            rem_vec = np.array([0,1,0,0,1,1,1,1,1,1,1])*1
        else:
            rem_vec = np.array([1,1,1,1,1,1,1,1,1,1,1])*1
        #rem_vec = [1,1,1,1,1,1]
        ttrc,vtrc = model.train((tx,ty),val_list=val_ds,rem_vec=rem_vec,log=True,use_ewc=use_ewc)
        for j,val_trace in enumerate(vtrc):
            t_itn = itn
            print 'Test on task '+str(j)+' '+str(val_trace[-1])#,str(val_trace[0])
            for v in val_trace:
                plots[j].append((t_itn,v))
                t_itn += 1
        itn = t_itn
        model.epch += 300
    #print len(plots)
    #for p in plots:
    #    print p
    for p,vacc_curve in enumerate(plots):
        if len(vacc_curve) == 1:
            plt.scatter([vacc_curve[0][0]],[vacc_curve[0][1]],label='Task '+str(p),marker='o')
        else:
            t = zip(*vacc_curve)
            plt.plot(*t,label='Task '+str(p),linestyle='-')#,marker='o')
    axes = plt.gca()
    axes.set_ylim([0,1])
    #axes.set_xlim([0,n_tasks+1])
    plt.legend()
    plt.ylabel('Validation accuracy')
    plt.xlabel('Iterations')
    plt.show()
    tf.reset_default_graph()

ds,vds = mnist.load()
x,y = map(np.float32,zip(*ds))
vx,vy = map(np.float32,zip(*vds))
print x.shape,y.shape,'--',vx.shape,vy.shape

num_tasks = 10
for use_ewc in [True,False]:
    print '*****','Tasks:',num_tasks,'EWC:',use_ewc,'*****'
    run_permute_exp((x,y),(vx,vy),num_tasks,use_ewc=use_ewc)
