import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import ewc_fg
import mnist
import tasks

def exp_ewc_all((x,y),(vx,vy),n_tasks,use_ewc=False,use_dropout=False):
    #sess = start_sess()
    plots = []
    model = ewc_fg.FullyConn()
    if use_dropout == False:
        model.dropout = 0
    else:
        model.dropout = 0.5
    print 'Trask 0: simple MNIST...'
    model.train((x,y),np.ones(6),val=(vx,vy),use_ewc=use_ewc)
    _,vacc = model.test(vx,y=vy)
    print 'Test on task 0',str(vacc)

    #Permutation tasks
    val_ds = [(vx,vy)]
    plots.append([(0,vacc)])
    accum_val = (list(vx),list(vy))
    for i in range(n_tasks):
        #if i == 2:
        #    model.lmbda = 0
        #else:
        #    model.lmbda = 400
        tds,tvds = tasks.permute(ds,vds)
        tx,ty = map(np.float32,zip(*tds))
        tvx,tvy = map(np.float32,zip(*tvds))
        val_ds.append((tvx,tvy))
        plots.append([])
        print 'Training on task '+str(i+1)+'...'
        if i == 3:
            rem_vec = [1,1,1,1,1,1]
        else:
            rem_vec = [1,1,1,1,1,1]
        #rem_vec = [1,1,1,1,1,1]
        model.train((tx,ty),val=(tvx,tvy),rem_vec=rem_vec,log=True,use_ewc=use_ewc)
        for j,(ovx,ovy) in enumerate(val_ds):
            _,vacc = model.test(ovx,y=ovy)
            print 'Test on task '+str(j)+' '+str(vacc)
            plots[j].append((i+1,vacc))
    for p,vacc_curve in enumerate(plots):
        if len(vacc_curve) == 1:
            plt.scatter([vacc_curve[0][0]],[vacc_curve[0][1]],label='Task '+str(p),marker='o')
        else:
            t = zip(*vacc_curve)
            plt.plot(*t,label='Task '+str(p),linestyle='-',marker='o')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,n_tasks+1])
    plt.legend()
    plt.ylabel('Validation accuracy')
    plt.xlabel('Task number')
    plt.show()
    tf.reset_default_graph()

ds,vds = mnist.load()
x,y = map(np.float32,zip(*ds))
vx,vy = map(np.float32,zip(*vds))
print x.shape,y.shape,'--',vx.shape,vy.shape

num_tasks = 35
for use_ewc in [True,False]:
    use_dropout = True
    print '*****','Tasks:',num_tasks,'EWC:',use_ewc,'Dropout:',use_dropout,'*****'
    exp_ewc_all((x,y),(vx,vy),num_tasks,use_ewc=use_ewc,use_dropout=use_dropout)
