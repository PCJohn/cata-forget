import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import ewc_fg
import mnist

def load_split_tasks(n):
    tc = 3000
    vc = 1000
    nclass = 10
    ds,vds = mnist.load(shuffle=False,train_count=tc,val_count=vc)
    sds = []
    for c in range(nclass):
        sds.append((ds[tc*c:tc*(c+1)],vds[vc*c:vc*(c+1)]))
    np.random.shuffle(sds)
    tasks = []
    for g in range(0,nclass,n):
        cds = sds[g:g+n]
        if len(cds) == n:
            ct,cv = zip(*cds)
            ct = np.array([i for j in ct for i in j])
            cv = np.array([i for j in cv for i in j])
            print ct.shape,cv.shape
            tasks.append((ct,cv))
            x,y = map(np.array,zip(*ct))
    return tasks

def run_split_mnist_exp(tasks,use_ewc=False):
    n_tasks = len(tasks)
    plots = []
    model = ewc_fg.FullyConn()
    #print 'Trask 0: simple MNIST...'
    #rem_vec = np.array([1,1,1,1,1,1,1])
    #model.train((x,y),rem_vec,val=(vx,vy),use_ewc=use_ewc)
    #_,vacc = model.test(vx,y=vy)
    #print 'Test on task 0',str(vacc)

    #val_ds = [(vx,vy)]
    val_ds = []
    #plots.append([(0,vacc)])
    #accum_val = (list(vx),list(vy))
    for i,(tds,tvds) in enumerate(tasks):
        tx,ty = map(np.float32,zip(*tds))
        tvx,tvy = map(np.float32,zip(*tvds))
        val_ds.append((tvx,tvy))
        print set(ty.argmax(axis=1))
        plots.append([])
        print 'Training on task '+str(i+1)+'...'
        if i >= 2:
            rem_vec = np.array([1,1,1,1,1,1,1])*1
        else:
            rem_vec = np.array([1,1,1,1,1,1,1])*1
        rem_vec = np.array([1,1,1,1,1,1])
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

#Create split MNIST classes
tasks = load_split_tasks(2)
for use_ewc in [True,False]:
    #print '*****','Tasks:',num_tasks,'EWC:',use_ewc,'*****'
    run_split_mnist_exp(tasks,use_ewc=use_ewc)
