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

def acc(py,y):
    return np.mean(py.argmax(axis=1)==y.argmax(axis=1))

current_milli_time = lambda: int(round(time.time()*1000))

class FullyConn():
    def __init__(self):
        self.lr = 1e-3
        self.eps = 1e-8
        self.bz = 200
        self.epch = 1200
        self.dropout = 0.5 #Actually 1-dropout
        self.lmbda = 200
 
        self.val_freq = 50
        self.log_freq = 100
        
        self.sess = None #start_sess()
        self.x = tf.placeholder(tf.float32,shape=[None,1,28,28])
        self.y = tf.placeholder(tf.float32,shape=[None,10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        #self.rem = tf.placeholder(tf.float32,shape=[6])

        d = 28*28*1
        out = tf.reshape(self.x,[-1,d])
        #out = tf.nn.dropout(out,0.8)

        H = 50
        w1 = tf.get_variable('w1',shape=[d,H],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',shape=[H],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,self.keep_prob)
        
        w2 = tf.get_variable('w2',shape=[H,H],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2',shape=[H],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w2)+b2)
        out = tf.nn.dropout(out,self.keep_prob)
        
        w3 = tf.get_variable('w3',shape=[H,10],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3',shape=[10],initializer=tf.zeros_initializer())
        self.out = tf.matmul(out,w3)+b3

        self.params = [w1,b1,w2,b2,w3,b3]
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out))
        self.ewc_loss = tf.Variable(0,tf.float32) #tf.constant(0.0,shape=[])
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon=self.eps).minimize(self.loss)
        self.n_tasks = 0

        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.params]
        self.anchor = [np.zeros(v.get_shape().as_list()) for v in self.params]
        self.fish_ph = [tf.placeholder(tf.float32,shape=v.get_shape().as_list()) for v in self.params]
        self.anch_ph = [tf.placeholder(tf.float32,shape=v.get_shape().as_list()) for v in self.params]
        #REM VEC here!
        #self.rem_vec = tf.placeholder(tf.float32,shape=(6,))
        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.params]
        self.task_F = []

        prob_list = tf.nn.softmax(self.out)
        class_index = tf.to_int32(tf.multinomial(tf.log(prob_list),1)[0][0])
        self.sample_loss = tf.log(prob_list[0,class_index])
        self.sample_grad = tf.gradients(self.sample_loss,self.params)
        
    #Update Fisher and anchors on new task
    def new_fish_anch(self,x,rem_vec,sample_count=1000):
        anchor_var_val = self.sess.run(self.params)
        self.anchor = [np.array(v,copy=True) for v in anchor_var_val]
        F = [np.zeros(v.get_shape().as_list()) for v in self.params]
        M = len(self.params)
        sample_ind = np.random.randint(0,x.shape[0],sample_count)
        for i in range(sample_count):
            sample = x[sample_ind[i]]
            grad = self.sess.run(self.sample_grad,feed_dict={self.x:np.array([sample]),self.keep_prob:1})
            for v in range(M):
                F[v] += np.square(grad[v])
        for v in range(M):
            F[v] = np.float32(F[v])/float(sample_count)
        
        task_F = []
        for v in range(M):
            task_F.append(F[v])
        self.task_F.append(task_F)

        """recon = True
        if recon == False:
            for v in range(M):
                self.fisher[v] += F[v]
        else:
            self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.params]
            print '>>>',rem_vec
            for r in range(self.n_tasks+1):
                for v in range(M):
                    self.fisher[v] += rem_vec[r]*self.task_F[r][v]
                    #self.fisher[v] = self.task_F[r][v]
                #print 'Task',r,'--',np.max(self.task_F[r][v])
        """
        
        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.params]
        for r in range(self.n_tasks+1):
            for v in range(M):
                self.fisher[v] += rem_vec[r]*self.task_F[r][v]
        self.n_tasks += 1
    
    def setup_ewc_loss(self):
        M = len(self.params)
        self.ewc_loss = tf.reduce_sum([0.5*
                                       self.lmbda*
                                       tf.reduce_sum(
                                       tf.multiply(self.fish_ph[v],
                                       tf.square(self.params[v]-self.anch_ph[v]))) 
                                       for v in range(M)])
        final_loss = tf.add(self.ewc_loss,self.loss)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(final_loss)
        k = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in set(self.sess.run(tf.report_uninitialized_variables()))])
        self.sess.run(k)

    def simple_hyperopt(self,x,y,vx,vy,rem_vec,use_ewc=False):
        saver = tf.train.Saver()
        tmp_file = 'tmp.ckpt'
        saver.save(self.sess,tmp_file)
        ind = range(x.shape[0])
        best = {'bz':100,'lr':1e-2,'epch':600,'val':0,'loss':999}
        lr_range = [1e-4,1e-3]
        r = 20
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
                epch_range = [1500,3000]
                epch = int(epch_range[0]+(epch_range[1]-epch_range[0])*np.random.random())
            e = np.random.random()
            if e < 0.5:
                bz = best['bz']
            else:
                bz = np.random.randint(15,300)
            for _ in range(epch):
                bi = np.random.choice(ind,bz)
                feed_dict={self.x:x[bi],self.y:y[bi],self.keep_prob:(1-self.dropout),self.learning_rate:lr}
                first_task = (self.n_tasks == 0)
                if use_ewc & (not first_task):
                    for v in range(len(self.params)):
                        feed_dict.update({self.fish_ph[v]:self.fisher[v]})
                        feed_dict.update({self.anch_ph[v]:self.anchor[v]})
                _,loss,eloss = self.sess.run([self.train_step,self.loss,self.ewc_loss],feed_dict=feed_dict)
                if np.isnan(loss):
                    lr_range[1] = lr
                    break
            pvy = self.sess.run(self.out,feed_dict={self.x:vx,self.keep_prob:1})
            val = acc(pvy,vy)
            print lr,bz,epch,val,loss,self.lmbda/2.,eloss,eloss-loss
            if val >  best['val']:
                best['val'] = val
                best['loss'] = loss
                best['bz'] = bz
                best['lr'] = lr
                best['epch'] = epch
            r -= 1
            if r == 0:
                if np.isnan(best['loss']):
                    r = 40
                    continue
        print 'Hyperparam config:',best
        self.lr = best['lr']
        self.bz = best['bz']
        self.epch = best['epch']
        saver.restore(self.sess,tmp_file)
        os.remove(tmp_file+'.meta')
        os.remove(tmp_file+'.index')
        os.remove(tmp_file+'.data-00000-of-00001')

    def train(self,train,rem_vec,val_list=[],log=True,use_ewc=False):
        if self.sess == None:
            self.sess = start_sess()
            self.sess.run(tf.global_variables_initializer())
        x,y = train
        #if len(val) > 0:
        #    vx,vy = val
        ind = range(x.shape[0])
        if log == True:
            print 'Hyperparameter search...'
        #self.simple_hyperopt(x,y,vx,vy,rem_vec,use_ewc=use_ewc)
        if log == True:
            print 'Training...'
        first_task = (self.n_tasks == 0)
        train_trace = []
        val_trace = [[] for _ in val_list]
        for epoch in range(self.epch):
            bi = np.random.choice(ind,self.bz) #batch index
            feed_dict = {self.x:x[bi],self.y:y[bi],self.keep_prob:(1-self.dropout),self.learning_rate:self.lr}
            #feed_dict.update({self.rem:rem_vec})
            if use_ewc & (not first_task):
                for v in range(len(self.params)):
                    feed_dict.update({self.fish_ph[v]:self.fisher[v]})
                    feed_dict.update({self.anch_ph[v]:self.anchor[v]})
            _,loss,eloss,py = self.sess.run([self.train_step,self.loss,self.ewc_loss,self.out],feed_dict=feed_dict)
            if epoch%self.val_freq == 0:
                for j,(v_x,v_y) in enumerate(val_list):
                    pvy = self.sess.run(self.out,feed_dict={self.x:v_x,self.keep_prob:1})
                    vacc = acc(pvy,v_y)
                    val_trace[j].append(vacc)
            if epoch%self.log_freq == 0:
                if log == True:
                    print 'Epoch',str(epoch)+': \tloss '+str(loss)+'\tewc '+str(eloss)+' \tval acc '+str(vacc)#str(acc(pvy,vy))
        #self.epch += 350
        if use_ewc == True:
            if first_task:
                self.setup_ewc_loss()
            self.new_fish_anch(val_list[-1][0],rem_vec)
        return (train_trace,val_trace)
        #end of train

    def test(self,x,y=None):
        py = self.sess.run(self.out,feed_dict={self.x:x,self.keep_prob:1})
        if y is None:
            return py
        else:
            return (py,acc(py,y))
