import os
import cv2
import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.misc import imread,imresize

import ewc_fg
import mnist
import feat_vgg
from feat_vgg import FeatureExtractor

path_to_dataset_folder = '/home/prithvi/dsets/MSCeleb/MSCeleb/'
path_to_saved_features = '/home/prithvi/dsets/MSCeleb/features.npy'

def load_msceleb(n_id=1000,shuffle=True,train_count=3000,val_count=1000):
    subd = os.listdir(path)
    subd.sort()
    subd = subd[21:]
    print subd[:30]
    idc = 0
    feat_path = path_to_saved_features
    feat = np.load(feat_path).item()
    tc = train_count
    vc = val_count
    ds = []
    vds = []
    for p in subd[:n_id]:
        pth = os.path.join(path,p)
        img_list = os.listdir(pth)
        np.random.shuffle(img_list)
        lab = np.zeros(n_id)
        lab[idc] = 1
        for img in img_list[:tc]:
            img_path = os.path.join(pth,img)
            im = imread(img_path,mode='RGB')
            im = imresize(im,disp_size)
            ds.append((feat[img_path][0,:],lab,im))
        for img in img_list[tc:tc+vc]:
            img_path = os.path.join(pth,img)
            im = imread(img_path,mode='RGB')
            im = imresize(im,disp_size)
            vds.append((feat[img_path][0,:],lab,im))
        idc += 1
    #Display a few samples
    for x,y,im in ds[:5]+ds[100:105]:
        plt.title(str(y)+'--'+str(x.shape))
        plt.imshow(im)
        plt.show()
    return (ds,vds)


def load_split_tasks(n,nppl):
    tc = 40
    vc = 50
    nclass = nppl
    ds,vds = load_msceleb(n_id=nclass,shuffle=False,train_count=tc,val_count=vc)
    sds = []
    for c in range(nclass):
        sds.append((ds[tc*c:tc*(c+1)],vds[vc*c:vc*(c+1)]))
    tasks = []
    for g in range(0,nclass,n):
        cds = sds[g:g+n]
        if len(cds) == n:
            ct,cv = zip(*cds)
            ct = [i for j in ct for i in j]
            cv = [i for j in cv for i in j]
            np.random.shuffle(ct)
            np.random.shuffle(cv)
            ctx,cty,cti = zip(*ct)
            cvx,cvy,cvi = zip(*cv)
            ts = set(np.array(cty).argmax(axis=1))
            ind = list(ts)
            ind.sort()
            cty = np.array(cty)[:,ind]
            cvy = np.array(cvy)[:,ind]
            tasks.append(((ctx,cty,cti),(cvx,cvy,cvi)))
    return tasks

def run_split_exp(arch,tasks,use_ewc=False,use_forget=False,hypopt=False,show_fish=[],repeat={},lmbda=200,pfrac=1):
    plots = []
    train_plots = []
    model = ewc_fg.FullyConn(arch)
    model.lr = 1e-4
    model.epch = 300
    model.bz = 150
    model.lmbda = 4e6
    model.log_freq = 5
    model.val_freq = 5
    model.train_freq = 5

    ttrc,vtrc = [],[]
    #Permutation tasks
    val_ds = []
    train_ds = []
    vitn = 0
    titn = 0
    plots = [[] for _ in range(n_tasks+1)]
    train_plots = [[] for _ in range(n_tasks+1)]
    fin_test_plot = []
    corr = []
    if len(vtrc) > 0:
        for v in vtrc[0]:
            plots[0].append((vitn*model.val_freq,v))
            vitn += 1
    if len(ttrc) > 0:
        for t in ttrc:
            train_plots[0].append((titn*model.train_freq,t))
            titn += 1
    
    rem_vec = 1*np.ones(len(tasks))
    env = 0
    K = 4
    env2task = range(K)
    env_acc = []
    task2env = [0 for _ in tasks]
    for i in range(len(tasks)):
        tds,tvds = tasks[i]
        tx,ty,ti = map(np.float32,tds)
        tvx,tvy,tvi = map(np.float32,tvds)
        val_ds.append((tvx,tvy))
        train_ds.append((tx,ty))
        print 'Training on task '+str(i+1)+' with data of shape:',tx.shape,ty.shape
        if i >= K:
            env = np.random.choice(range(K))
            print 'People in environment',env,'changed'
            if use_forget == True:
                rem_vec[env2task[env]] = 0
            model.epch += 50
            env2task[env] = i
            task2env[i] = env
            print 'Learning task',i,'in environment',env
        else:
            env = i
        print env2task,rem_vec[:i]
        if use_forget == True:
            head = env
        else:
            head = env #i
        ttrc,vtrc = model.train((tx,ty),head,val_list=val_ds,rem_vec=rem_vec,log=True,use_ewc=use_ewc,hypopt=hypopt)
        
        if i in show_fish:
            F = model.fisher[2*1]
            plt.imshow(F,cmap='OrRd')
            plt.show()
       
        #Save Fisher for correlation
        corr.append([])
        if use_ewc == True:
            for f in model.F:
                corr[-1].append(f.copy())
        
        t_itn = titn
        for t in ttrc:
            train_plots[i+1].append((t_itn*model.train_freq,t))
            t_itn += 1
        titn = t_itn

        m_ac = []
        for j,val_trace in enumerate(vtrc):
            t_itn = vitn
            print 'Test on task '+str(j)+' '+str(val_trace[-1])#,str(val_trace[0])
            
            if j in env2task:
                m_ac.append(val_trace[-1])

            for v in val_trace:
                if j in [0,5,15,20]:
                    plots[j].append((t_itn*model.val_freq,v))
                t_itn += 1
            if i == (n_tasks-1):
                fin_test_plot.append((j,val_trace[-1]))
        vitn = t_itn
        env_acc.append((i,np.mean(m_ac),min(m_ac)))

    #Task similarity
    #task_sim_mat(val_ds)
    #Show task correlations
    if use_ewc == True:
        print 'Showing corrletaion matrices...'
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
            plt.title('Layer '+str(lyr))
            plt.imshow(m,cmap='OrRd',interpolation='none')
            plt.show()
    ls = ['p','*','^','s','o']
    
    #Show acc on env
    t = zip(*env_acc)
    print '>>>',t
    plt.plot(t[0],t[1],markersize=8,marker='s')
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.set_xlim([0,n_tasks+1])
    plt.ylabel('Accuracy on active tasks')
    plt.xlabel('Task number')
    plt.show()
    
    #Show training plots
    for p,train_curve in enumerate(train_plots):
        t = zip(*train_curve)
        plt.plot(*t,label='Task '+str(p),linestyle='-')
    axes = plt.gca()
    axes.set_ylim([0,1])
    plt.legend(loc=4)
    plt.ylabel('Training accuracy')
    plt.xlabel('Iterations')
    plt.show()
    
    #Show validation plots
    for p,vacc_curve in enumerate(plots):
        if len(vacc_curve) == 0:
            continue
        if len(vacc_curve) == 1:
            plt.scatter([vacc_curve[0][0]],[vacc_curve[0][1]],label='Task '+str(p),marker='o')
        else:
            t = zip(*vacc_curve)
            plt.plot(*t,label='Task '+str(p),linestyle='-')#,marker='o')
    axes = plt.gca()
    axes.set_ylim([0,1])
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
    plt.xlabel('Task number')
    plt.ylabel('Final validation accuracy')
    plt.show()
    tf.reset_default_graph()

#####***** EXPERIMENT CONTROLS HERE *****#####
if sys.argv[1] == 'feat':
    feat_vgg.extract_feat()

elif sys.argv[1] == 'exp':
    path = path_to_dataset_folder
    disp_size = (64,64)
    
    n_classes_per_task = 10
    n_people = 250
    n_head = n_people/n_classes_per_task

    arch = {'in_dim':4096,
        'hid':[400,400],
        'n_out':n_classes_per_task,
        'n_hid':2,
        'n_head':n_head}

    n_tasks = n_people/n_classes_per_task
    num_tasks = n_tasks
    tasks = load_split_tasks(n_classes_per_task,n_people)

    use_hypopt = False
    show_fish = [] #[0,5,10,14]
    repeat = {} #{1:0,5:0}
    for use_ewc in [True,False]:
        for use_forget in [True,False]:
            print '\t*****','Tasks:',num_tasks,'EWC:',use_ewc,'Use Forget:',use_forget,'*****'
            lmbda = 1000
            run_split_exp(arch,tasks,
                            use_ewc=use_ewc,
                            use_forget=use_forget,
                            hypopt=use_hypopt,
                            show_fish=show_fish,
                            repeat=repeat,
                            lmbda=lmbda)
