# Experiment to demonstrate selective forgetting with permuted MNIST images.
#
# Usage:
# 1. Download the MNIST training set folder (trainingSet.tar.gz) from: https://www.kaggle.com/scolianni/mnistasjpg
# 2. Point to the folder trainingSet (line 6 in mnist.py)
# 3. Run: python mnist_permute_exp.py
#
# To change the forget policy, edit the block in lines 95-105
#
# To modify the architecture, edit lines 210-214 
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import ewc_fg
import mnist

#Permute pixels of MNIST images
def permute(ds,vds,pfrac=1):
    ap = range(28*28)
    if pfrac < 1:
        l = int(len(ap)*pfrac)
        t1,t2 = ap[:l],ap[l:]
        np.random.shuffle(t1)
        ap = t1+t2
    else:
        np.random.shuffle(ap)
    tds,tvds = [],[]
    for l,(x,y) in enumerate(ds):
        tds.append((x[ap],y))
    for vx,vy in vds:
        tvds.append((vx[ap],vy))
    return ((ap,ap),tds,tvds)

#Find the similarity between the permuted MNIST tasks
def task_sim_mat(val_ds):
    n = len(val_ds)
    m = np.zeros((n,n))
    nsmpl = 1000
    vind = range(len(val_ds[0][0]))
    for t1,(t1x,t1y) in enumerate(val_ds):
        for t2,(t2x,t2y) in enumerate(val_ds):
            for _ in range(nsmpl):
                rind = np.random.choice(vind)
                rx1,rx2 = t1x[rind],t2x[rind]
                m[t1,t2] += np.linalg.norm(rx1-rx2)
            m[t1,t2] /= float(nsmpl)
    print '>>>>',m

#Method to run the experiment and plot results
def run_permute_exp(arch,(x,y),(vx,vy),n_tasks,use_ewc=False,use_forget=False,hypopt=False,show_fish=[],repeat={},lmbda=200,pfrac=1):
    plots = []
    train_plots = []
    model = ewc_fg.FullyConn(arch)
    model.lmbda = lmbda

    head = 0
    print 'Trask 0: simple MNIST...'
    #rem_vec = np.array([1,1,1,1,1,1,1,1,1,1,1])
    rem_vec = np.ones(n_tasks+1)
    ttrc,vtrc = model.train((x,y),head,rem_vec,val_list=[(vx,vy)],hypopt=hypopt,log=True,use_ewc=use_ewc)
    print 'Test on task 0',str(vtrc[0][-1])

    #Permutation tasks
    val_ds = [(vx,vy)]
    train_ds = [(x,y)]
    vitn = 0
    titn = 0
    plots = [[] for _ in range(n_tasks+1)]
    train_plots = [[] for _ in range(n_tasks+1)]
    fin_test_plot = []
    corr = []
    for v in vtrc[0]:
        plots[0].append((vitn*model.val_freq,v))
        vitn += 1
    for t in ttrc:
        train_plots[0].append((titn*model.train_freq,t))
        titn += 1
    for i in range(n_tasks):
        if i in repeat.keys():
            print 'Task',(i+1),'is a repeat of task',repeat[i]
            (tx,ty),(tvx,tvy) = train_ds[repeat[i]],val_ds[repeat[i]]
        else:
            (rp,cp),tds,tvds = permute(ds,vds,pfrac=pfrac)
            tx,ty = map(np.float32,zip(*tds))
            tvx,tvy = map(np.float32,zip(*tvds))
        val_ds.append((tvx,tvy))
        train_ds.append((x,y))

        print 'Training on task '+str(i+1)+'...'
        
        ##### Forget policy here #####
        #rem_vec = 1*np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        rem_vec = np.ones(n_tasks+1)
        if use_forget == True:
            if i >= 4: # from task 4 onwards, forget tasks 0,1,3,5
                rem_vec[0] = 0
                rem_vec[1] = 0
                rem_vec[3] = 0
                rem_vec[5] = 0
            print 'Remember vector:',rem_vec
        ##############################

        ttrc,vtrc = model.train((tx,ty),head,val_list=val_ds,rem_vec=rem_vec,log=True,use_ewc=use_ewc,hypopt=hypopt)
        
        #Display the Fisher matrices for the selected tasks
        if i in show_fish:
            F = model.fisher[2*1]
            print('\nDisplaying the Fisher Information Matrix for task: '+str(i))
            plt.imshow(F,cmap='OrRd')
            plt.title('Fisher Information Matrix for task: '+str(i))
            plt.show()
       
        #Save Fisher for the correlation matrices
        corr.append([])
        if use_ewc == True:
            for f in model.F:
                corr[-1].append(f.copy())
        
        #Append plt data
        t_itn = titn
        for t in ttrc:
            train_plots[i+1].append((t_itn*model.train_freq,t))
            t_itn += 1
        titn = t_itn
        for j,val_trace in enumerate(vtrc):
            t_itn = vitn
            print 'Test on task '+str(j)+' '+str(val_trace[-1])
            for v in val_trace:
                plots[j].append((t_itn*model.val_freq,v))
                t_itn += 1
            if i == (n_tasks-1):
                fin_test_plot.append((j,val_trace[-1]))
        vitn = t_itn
    
    #Show how orthogonal the task dependent Fisher matrices are
    if use_ewc == True:
        print 'Showing correlation matrices...'
        corr_mat = [np.zeros((n_tasks,n_tasks)) for _ in range(2)]
        wt = [2]
        for t1 in range(n_tasks):
            for t2 in range(n_tasks):
                for w,lyr in enumerate(wt):
                    f1 = corr[t1][lyr]
                    f2 = corr[t2][lyr]
                    f1 = (f1/np.linalg.norm(f1)).flatten()
                    f2 = (f2/np.linalg.norm(f2)).flatten()
                    cx = f1.T.dot(f2)
                    corr_mat[w][t1,t2] = cx
        for w,lyr in enumerate(wt):
            m = corr_mat[w]
            plt.title('Inter-task correlation for layer '+str(lyr)+' weights')
            plt.imshow(m,cmap='OrRd',interpolation='none')
            plt.show()
    
    #Show training plots
    for p,train_curve in enumerate(train_plots):
        t = zip(*train_curve)
        plt.plot(*t,label='Task '+str(p),linestyle='-')
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.title('Training plots')
    plt.legend(loc=4)
    plt.ylabel('Training accuracy')
    plt.xlabel('Iterations')
    plt.show()
    
    #Show validation plots
    for p,vacc_curve in enumerate(plots):
        if len(vacc_curve) == 1:
            plt.scatter([vacc_curve[0][0]],[vacc_curve[0][1]],label='Task '+str(p),marker='o')
        else:
            t = zip(*vacc_curve)
            plt.plot(*t,label='Task '+str(p),linestyle='-')
    axes = plt.gca()
    axes.set_ylim([0,1])
    if use_forget == True:
        plt.title('Validation with the selective forgetting policy')
    else:
        plt.title('Validation without forgetting (remember all tasks)')
    plt.legend(bbox_to_anchor=(1.1,0.3))
    plt.ylabel('Validation accuracy')
    plt.xlabel('Iterations')
    plt.show()
   
    #Show final validation accuracy variation
    t = zip(*fin_test_plot)
    plt.plot(*t,label='Val curve',marker='s',color='b',linewidth=0,markersize=10)
    axes = plt.gca()
    axes.set_xlim([0,n_tasks])
    axes.set_ylim([0,1])
    if use_forget == True:
        plt.title('Final validation with the selective forgetting policy')
    else:
        plt.title('Final validation without forgetting (remember all tasks)')
    plt.xlabel('Task number')
    plt.ylabel('Final validation accuracy')
    plt.show()
    tf.reset_default_graph()

#####***** EXPERIMENT CONTROLS HERE *****#####
ds,vds = mnist.load(flatten=True)
x,y = map(np.float32,zip(*ds))
vx,vy = map(np.float32,zip(*vds))
print 'Loaded ds shape:',x.shape,y.shape
print 'Loaded val ds shape:',vx.shape,vy.shape
arch = {'in_dim':28*28, # input shape
        'hid':[100,100],# hidden layer dimensions
        'n_out':10,     # number of output units
        'n_hid':2,      # number of hidden layers
        'n_head':1}     # number of output heads
fraction_to_permute = 1 #Fraction of pixels in images to permute (to generate a new task)
num_tasks = 10          #Number of tasks
use_hypopt = False      #Use hyperparameter optimization for every new task
show_fish = [0,5]       #Visualize the Fisher info matrix for these tasks
repeat = {} #{1:0,5:0}  #{a:b} will mean tasks a will be a copy of task b
lmbda = 200             #EWC scaling

for use_ewc in [True,False]: # whether or not to use EWC
    if use_ewc == True:
        for use_forget in [True,False]: # whether or not to use the selective forget policy
            print '\t*****','Tasks:',num_tasks,'| EWC:',use_ewc,'| Use forget:',use_forget,'*****'
            run_permute_exp(arch,(x,y),(vx,vy),num_tasks,
                            use_ewc=use_ewc,
                            use_forget=use_forget,
                            hypopt=use_hypopt,
                            show_fish=show_fish,
                            repeat=repeat,
                            lmbda=lmbda,
                            pfrac=fraction_to_permute)
