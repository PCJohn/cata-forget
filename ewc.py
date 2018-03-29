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
        self.epch = 1300
        self.dropout = 0.5 #Actually 1-dropout
        self.lmbda = 400
 
        self.sess = None #start_sess()
        self.x = tf.placeholder(tf.float32,shape=[None,1,28,28])
        self.y = tf.placeholder(tf.float32,shape=[None,10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.ewc_lambda = tf.placeholder(tf.float32)

        d = 28*28*1
        out = tf.reshape(self.x,[-1,d])
        #out = tf.nn.dropout(out,0.8)

        w1 = tf.get_variable('w1',shape=[d,400],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',shape=[400],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,self.keep_prob)
        
        w2 = tf.get_variable('w2',shape=[400,400],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2',shape=[400],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w2)+b2)
        out = tf.nn.dropout(out,self.keep_prob)
        
        w3 = tf.get_variable('w3',shape=[400,10],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3',shape=[10],initializer=tf.zeros_initializer())
        self.out = tf.matmul(out,w3)+b3

        self.params = [w1,b1,w2,b2,w3,b3]
        #self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=self.out))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=self.out))
        self.ewc_loss = tf.Variable(0,tf.float32) #tf.constant(0.0,shape=[])
        #final_loss = 0.5*self.lmbda*self.ewc_loss+self.loss
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate,epsilon=self.eps).minimize(self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(self.loss)
        self.fisher = None
        self.n_tasks = 0

        prob_list = tf.nn.softmax(self.out)
        class_index = tf.to_int32(tf.multinomial(tf.log(prob_list),1)[0][0])
        self.sample_loss = tf.log(prob_list[0,class_index])
        self.sample_grad = tf.gradients(self.sample_loss,self.params)
        
    def augment_loss_ewc(self,x,sample_count=1000):
        anchor_var_val = self.sess.run(self.params)
        anchor = [np.array(v,copy=True) for v in anchor_var_val]
        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.params]
        
        M = len(self.params)
        sample_ind = np.random.randint(0,x.shape[0],sample_count)
        for i in range(sample_count):
            sample = x[sample_ind[i]]
            grad = self.sess.run(self.sample_grad,feed_dict={self.x:np.array([sample]),self.keep_prob:1})
            for v in range(M):
                self.fisher[v] += np.square(grad[v])
        if self.n_tasks == 0:
            self.ewc_loss = 0
        #self.ewc_loss = 0
        for v in range(M):
            self.fisher[v] = self.fisher[v]/float(sample_count)
            self.ewc_loss += 0.5*self.lmbda*tf.reduce_sum(tf.multiply(np.float32(self.fisher[v]),tf.square(self.params[v]-anchor[v])))
        final_loss = tf.add(self.ewc_loss,self.loss)
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(final_loss)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.lr).minimize(final_loss)
        k = tf.variables_initializer(
            [v for v in tf.global_variables() if v.name.split(':')[0] in set(self.sess.run(tf.report_uninitialized_variables()))
            ])
        self.sess.run(k)

        del anchor
        self.n_tasks += 1

    def simple_hyperopt(self,x,y,vx,vy,use_ewc=False):
        saver = tf.train.Saver()
        tmp_file = 'tmp.ckpt'
        saver.save(self.sess,tmp_file)
        ind = range(x.shape[0])
        best = {'bz':100,'lr':1e-2,'epch':600,'val':0,'loss':999}
        lr_range = [1e-4,1e-2]
        r = 40
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
                epch_range = [1000,6000]
                epch = int(epch_range[0]+(epch_range[1]-epch_range[0])*np.random.random())
            e = np.random.random()
            if e < 0.5:
                bz = best['bz']
            else:
                bz = np.random.randint(15,300)
            for _ in range(epch):
                bi = np.random.choice(ind,bz)
                _,loss,eloss = self.sess.run([self.train_step,self.loss,self.ewc_loss],feed_dict={self.x:x[bi],self.y:y[bi],
                                                                   self.keep_prob:(1-self.dropout),self.learning_rate:lr})
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

    def train(self,train,val=None,log=True,use_ewc=False):
        if self.sess == None:
            self.sess = start_sess()
            self.sess.run(tf.global_variables_initializer())
        x,y = train
        if val != None:
            vx,vy = val
        ind = range(x.shape[0])
        if log == True:
            print 'Hyperparameter search...'
        #self.simple_hyperopt(x,y,vx,vy,use_ewc=use_ewc)
        if log == True:
            print 'Training...'
        for epoch in range(self.epch):
            bi = np.random.choice(ind,self.bz) #batch index
            _,loss,eloss,py = self.sess.run([self.train_step,self.loss,self.ewc_loss,self.out],feed_dict={self.x:x[bi],
                                        self.y:y[bi],self.keep_prob:(1-self.dropout),self.learning_rate:self.lr})
            if epoch%100 == 0:
                pvy = self.sess.run(self.out,feed_dict={self.x:vx,self.keep_prob:1})
                if log == True:
                    print 'Epoch',str(epoch)+': \tloss '+str(loss)+'\tewc '+str(eloss)+' \tval acc '+str(acc(pvy,vy))
        
        if use_ewc == True:
            self.augment_loss_ewc(vx)
        
        #end of train

    def test(self,x,y=None):
        py = self.sess.run(self.out,feed_dict={self.x:x,self.keep_prob:1})
        if y is None:
            return py
        else:
            return (py,acc(py,y))

"""#(ds,vds),(pds_list,pvds_list) = mnist_permute.load(permute=1)
ds,vds = mnist.load()
x,y = map(np.float32,zip(*ds))
vx,vy = map(np.float32,zip(*vds))
print x.shape,y.shape,'--',vx.shape,vy.shape

#EXPERIMENT
sess = start_sess()
model = FullyConn(sess)
print 'Trask 0: simple MNIST...'
model.train((x,y),val=(vx,vy))
_,vacc = model.test(vx,y=vy)
print 'Test on task 0',str(vacc)
model.augment_loss_ewc(x,lmbda=400)

#Permutation tasks
val_ds = [(vx,vy)]
for i in range(6):
    tds,tvds = tasks.permute(ds,vds)
    tx,ty = map(np.float32,zip(*tds))
    tvx,tvy = map(np.float32,zip(*tvds))
    val_ds.append((tvx,tvy))

    print 'Training on task '+str(i+1)+'...'
    #model.epch += 100
    model.train((tx,ty),val=(tvx,tvy))
    for j,(ovx,ovy) in enumerate(val_ds):
        _,vacc = model.test(ovx,y=ovy)
        print 'Test on task '+str(j)+' '+str(vacc)
    #lmbda -= 200
    #lmbda = max(10,lmbda)
    model.augment_loss_ewc(tx,lmbda=30)

#TODO: Split tasks

sess.close()
"""
