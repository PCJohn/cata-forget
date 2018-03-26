import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import ewc
import mnist
import tasks

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    return sess

#def hyperopt(model,(tx,ty)):
#    sess = start_sess()
#    init_epch = model.epch
#    model.train
#    model.epch = init_epch

#EXPERIMENT
def exp_ewc_all((x,y),(vx,vy),n_tasks,use_ewc=False,use_dropout=False):
    #sess = start_sess()
    plots = []
    model = ewc.FullyConn()
    if use_dropout == False:
        model.dropout = 0
    else:
        model.dropout = 0.5
    print 'Trask 0: simple MNIST...'
    model.train((x,y),val=(vx,vy),use_ewc=use_ewc)
    _,vacc = model.test(vx,y=vy)
    print 'Test on task 0',str(vacc)
    #if use_ewc == True:
    #    model.augment_loss_ewc(x,lmbda=300) #Weird: Check. Normalize?

    #Permutation tasks
    val_ds = [(vx,vy)]
    plots.append([(0,vacc)])
    for i in range(n_tasks):
        tds,tvds = tasks.permute(ds,vds)
        tx,ty = map(np.float32,zip(*tds))
        tvx,tvy = map(np.float32,zip(*tvds))
        val_ds.append((tvx,tvy))
        plots.append([])
        print 'Training on task '+str(i+1)+'...'
        #lmbda = hyperopt(model,(tx,ty))
        #print 'Best lambda:',lmbda
        model.train((tx,ty),val=(tvx,tvy),log=True,use_ewc=use_ewc)
        for j,(ovx,ovy) in enumerate(val_ds):
            _,vacc = model.test(ovx,y=ovy)
            print 'Test on task '+str(j)+' '+str(vacc)
            plots[j].append((i+1,vacc))
        #if use_ewc == True:
        #    model.augment_loss_ewc(tx,lmbda=lmbda)
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
    plt.show()
    tf.reset_default_graph()
    #sess.close()

ds,vds = mnist.load()
x,y = map(np.float32,zip(*ds))
vx,vy = map(np.float32,zip(*vds))
print x.shape,y.shape,'--',vx.shape,vy.shape

#print '***** Plain *****'
#exp_ewc_all(2,use_ewc=False)
#print '***** EWC *****'
#exp_ewc_all(2,use_ewc=True)

num_tasks = 6 
for use_ewc in [True,False]:
    for use_dropout in [False,True]:
        print '*****','Tasks:',num_tasks,'EWC:',use_ewc,'Dropout:',use_dropout,'*****'
        #sess = start_sess()
        exp_ewc_all((x,y),(vx,vy),num_tasks,use_ewc=use_ewc,use_dropout=use_dropout)
        #tf.reset_default_graph()
        #sess.close()
