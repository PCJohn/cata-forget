# Module to implement Elastic Weight Consolidation
#
# Author: Prithvijit Chakrabarty (prithvichakra@gmail.com)

import os
import cv2
import time
import numpy as np
import random
import tensorflow as tf

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    return sess

#Compute accuracy
def acc(py,y):
    return np.mean(py.argmax(axis=1)==y.argmax(axis=1))

current_milli_time = lambda: int(round(time.time()*1000))

#Fully connected model with EWC
class FullyConn():
    def __init__(self,arch):
        #Default hyperparams
        self.lr = 1e-3
        self.eps = 1e-8
        self.bz = 100
        self.epch = 3000
        self.dropout = 0.5
        self.lmbda = 200
 
        #Logging params
        self.val_freq = 25
        self.train_freq = 25
        self.log_freq = 100
        
        self.in_dim = arch['in_dim']
        self.H = arch['hid']
        self.n_out = arch['n_out']
        self.n_hid = arch['n_hid']
        self.n_head = arch['n_head']

        self.multi_head = (self.n_head > 1)

        self.sess = None
        self.x = tf.placeholder(tf.float32,shape=[None,self.in_dim])
        self.y = tf.placeholder(tf.float32,shape=[None,self.n_out])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        d = self.in_dim
        out = tf.reshape(self.x,[-1,d])

        #First hidden layer
        w1 = tf.get_variable('w1',shape=[d,self.H[0]],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',shape=[self.H[0]],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,self.keep_prob)
        
        #Deeper hidden layers
        self.base_params = [w1,b1]
        for l in range(1,self.n_hid):
            fshape = self.H[l]
            if l == 1:
                fshape = self.H[0]
            w = tf.get_variable('w'+str(l+1),shape=[fshape,self.H[l]],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b'+str(l+1),shape=[self.H[l]],initializer=tf.zeros_initializer())
            self.base_params.append(w)
            self.base_params.append(b)
            out = tf.nn.relu(tf.matmul(out,w)+b)
            out = tf.nn.dropout(out,self.keep_prob)
        
        #Multiple outputs for multiple heads
        self.out = []
        self.out_params = []
        for l in range(self.n_head):
            w = tf.get_variable('w'+str(l+self.n_hid+1),shape=[self.H[-1],self.n_out],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b'+str(l+self.n_hid+1),shape=[self.n_out],initializer=tf.zeros_initializer())
            self.out_params.append(w)
            self.out_params.append(b)
            self.out.append(tf.matmul(out,w)+b)
        self.params = self.base_params+self.out_params
    
        #Use EWC throughout if only one head
        if not self.multi_head:
            self.base_params += self.out_params
        
        #Loss and train step per head
        self.loss = []
        self.train_step = []
        for l in range(self.n_head):
            self.loss.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out[l])))
            self.train_step.append(tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon=self.eps).minimize(self.loss[l]))
        self.ewc_loss = tf.Variable(0,tf.float32)
        
        self.n_tasks = 0
        self.task2head = {}

        #Init. fisher info. and anchor
        self.fisher = []
        self.fish_ph = []
        self.anchor = []
        self.anch_ph = []
        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.base_params]
        self.anchor = [np.zeros(v.get_shape().as_list()) for v in self.base_params]
        self.fish_ph = [tf.placeholder(tf.float32,shape=v.get_shape().as_list()) for v in self.base_params]
        self.anch_ph = [tf.placeholder(tf.float32,shape=v.get_shape().as_list()) for v in self.base_params]
        
        self.task_F = []

        self.sample_loss = []
        self.sample_grad = []
        for l in range(self.n_head):
            hparams = [k for k in self.base_params]

            if self.multi_head:
                hparams.append(self.out_params[2*l])
                hparams.append(self.out_params[2*l+1])

            prob_list = tf.nn.softmax(self.out[l])
            class_index = tf.to_int32(tf.multinomial(tf.log(prob_list),1)[0][0])
            self.sample_loss.append(tf.log(prob_list[0,class_index]))
            self.sample_grad.append(tf.gradients(self.sample_loss[l],hparams))
        
    #Update Fisher and anchors on new task
    def new_fish_anch(self,x,head,rem_vec,sample_count=1000):
        #Compute the anchor parameters (theta* in the EWC paper)
        hparams = self.base_params        
        anchor_var_val = self.sess.run(self.base_params)
        self.anchor = [np.array(v,copy=True) for v in anchor_var_val]
        #Compute the Fisher info. matrix
        F = [np.zeros(v.get_shape().as_list()) for v in self.base_params]#hparams]
        M = len(self.base_params)
        sample_ind = np.random.randint(0,x.shape[0],sample_count)
        for i in range(sample_count):
            sample = x[sample_ind[i]]
            grad = self.sess.run(self.sample_grad[head],feed_dict={self.x:np.array([sample]),self.keep_prob:1})
            for v in range(M):
                F[v] += np.square(grad[v])
        for v in range(M):
            F[v] = np.float32(F[v])/float(sample_count)
       
        #Task specific Fisher
        task_F = []
        for v in range(M):
            task_F.append(F[v])
        self.task_F.append(task_F) 
        
        #Weighted Fisher reconstruction for forgetting
        self.F = []
        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.base_params]
        for r in range(self.n_tasks+1):
            for v in range(M):
                self.F.append(F[v].copy())
                self.fisher[v] += rem_vec[r]*self.task_F[r][v]

    #Add graph nodes for EWC losses per head
    def setup_ewc_loss(self):
        final_loss = []
        self.train_step = []
        hparams = [k for k in self.base_params]
        M = len(hparams)
        self.ewc_loss = (tf.reduce_sum([0.5*
                                       self.lmbda*
                                       tf.reduce_sum(
                                       tf.multiply(self.fish_ph[v],
                                       tf.square(hparams[v]-self.anch_ph[v]))) 
                                       for v in range(M)])
                                    )
        for l in range(self.n_head):
            final_loss.append(tf.add(self.ewc_loss,self.loss[l]))
            self.train_step.append(tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(final_loss[l]))
        k = set(self.sess.run(tf.report_uninitialized_variables()))
        uninit = [v for v in tf.global_variables() if v.name.split(':')[0] in k]
        self.sess.run(tf.variables_initializer(uninit))

    #Hyperparameter optimizer
    def hyperopt(self,x,y,vx,vy,head,rem_vec,use_ewc=False):
        saver = tf.train.Saver()
        tmp_file = 'tmp.ckpt'
        saver.save(self.sess,tmp_file)
        ind = range(x.shape[0])
        #best = {'bz':100,'lr':1e-2,'epch':600,'val':0,'loss':999}
        best = {'bz':100,'lr':1e-3,'epch':100,'val':0,'loss':999,'tot_loss':999,'eloss':999}
        lr_range = [1e-5,1e-1]
        r = 50
        while r > 0:
            saver.restore(self.sess,tmp_file)
            e = np.random.random()
            if e < 0.5:
                lr = best['lr']
            else:
                lr = np.random.choice(np.arange(lr_range[0],lr_range[1],(lr_range[1]-lr_range[0])/100000.))
            e = np.random.random()
            if e < 0.5:
                epch = best['epch']
            else:
                epch_range = [30,100]
                epch = int(epch_range[0]+(epch_range[1]-epch_range[0])*np.random.random())
            e = np.random.random()
            if e < 0.5:
                bz = best['bz']
            else:
                bz = np.random.randint(5,200)
            for _ in range(epch):
                bi = np.random.choice(ind,bz)
                feed_dict={self.x:x[bi],self.y:y[bi],self.keep_prob:(1-self.dropout),self.learning_rate:lr}
                first_task = (self.n_tasks == 0)
                if use_ewc & (not first_task):
                    for v in range(len(self.base_params)):
                        feed_dict.update({self.fish_ph[v]:self.fisher[v]})
                        feed_dict.update({self.anch_ph[v]:self.anchor[v]})
                _,loss,eloss = self.sess.run([self.train_step[head],self.loss[head],self.ewc_loss],feed_dict=feed_dict)
                tot_loss = loss+eloss
                if np.isnan(loss):
                    lr_range[1] = lr
                    break
            pvy = self.sess.run(self.out[head],feed_dict={self.x:vx,self.keep_prob:1})
            val = acc(pvy,vy)
            
            if ((eloss<=best['eloss'])&(loss<=best['loss'])):
                best['loss'] = loss
                best['eloss'] = eloss
                best['tot_loss'] = tot_loss
                best['bz'] = bz
                best['lr'] = lr
                best['epch'] = epch+1
                best['val'] = val
            r -= 1
            if r == 0:
                if np.isnan(best['loss']):
                    r = 40
                    continue
            print 'lr',lr,'bz',bz,'epch',epch,'val',val,'L',loss,self.lmbda/2.,'EL',eloss,eloss-loss,'lmbda',self.lmbda
        
        print 'Hyperparam config:',best
        self.lr = best['lr']
        self.bz = best['bz']
        self.epch = best['epch']
        saver.restore(self.sess,tmp_file)
        os.remove('./'+tmp_file+'.meta')
        os.remove('./'+tmp_file+'.index')
        os.remove('./'+tmp_file+'.data-00000-of-00001')

    #Method to train the network
    def train(self,train,head,rem_vec,val_list=[],hypopt=False,log=True,use_ewc=False):
        self.task2head.update({self.n_tasks:head})
        if self.sess == None:
            self.sess = start_sess()
            self.sess.run(tf.global_variables_initializer())
        x,y = train
        ind = range(x.shape[0])
        if hypopt == True:
            if log == True:
                print 'Hyperparameter search...'
            vx,vy = val_list[-1] #Use val data of most recent task for hyperopt
            self.hyperopt(x,y,vx,vy,head,rem_vec,use_ewc=use_ewc)
        if log == True:
            print 'Training...'
        first_task = (self.n_tasks == 0)
        train_trace = []
        val_trace = [[] for _ in val_list]
        
        hparams = [k for k in self.base_params]
        
        if log == True:
            print 'Task2Head map:',self.task2head
        for epoch in range(self.epch):
            bi = np.random.choice(ind,self.bz) #batch index
            feed_dict = {self.x:x[bi],self.y:y[bi],self.keep_prob:(1-self.dropout),self.learning_rate:self.lr}
            if use_ewc & (not first_task):
                for v in range(len(hparams)):
                    feed_dict.update({self.fish_ph[v]:self.fisher[v]})
                    feed_dict.update({self.anch_ph[v]:self.anchor[v]})
            _,loss,eloss,py = self.sess.run([self.train_step[head],self.loss[head],self.ewc_loss,self.out[head]],feed_dict=feed_dict)
            if epoch%self.train_freq == 0:
                train_trace.append(acc(py,y[bi]))
            if epoch%self.val_freq == 0:
                for j,(v_x,v_y) in enumerate(val_list):
                    if self.multi_head:
                        task_head = self.task2head[j]
                    else:
                        task_head = head
                    pvy = self.sess.run(self.out[task_head],feed_dict={self.x:v_x,self.keep_prob:1})
                    vacc = acc(pvy,v_y)
                    val_trace[j].append(vacc)
            if epoch%self.log_freq == 0:
                if log == True:
                    print 'Epoch',str(epoch)+': \tloss '+str(loss)+'\tewc '+str(eloss)+' \tval acc '+str(vacc)
        if use_ewc == True:
            if first_task:
                self.setup_ewc_loss()
            self.new_fish_anch(val_list[-1][0],head,rem_vec)
        self.n_tasks += 1
        return (train_trace,val_trace)
        #end of train

    #Method to run the model
    def test(self,x,head,y=None):
        py = self.sess.run(self.out[head],feed_dict={self.x:x,self.keep_prob:1})
        if y is None:
            return py
        else:
            return (py,acc(py,y))
