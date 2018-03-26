import os
import cv2
import time
import numpy as np
import random
import tensorflow as tf

"""import mnist
import tasks
"""
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
        self.epch = 700
        self.dropout = 0.5 #Actually 1-dropout
        self.lmbda = 1000
 
        self.sess = None #start_sess()
        self.x = tf.placeholder(tf.float32,shape=[None,1,28,28])
        self.y = tf.placeholder(tf.float32,shape=[None,10])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)
        self.ewc_lambda = tf.placeholder(tf.float32)

        d = 28*28*1
        out = tf.reshape(self.x,[-1,d])
        
        #w1 = weight([d,2000])
        #b1 = bias([2000])
        w1 = tf.get_variable('w1',shape=[d,800],initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable('b1',shape=[800],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,self.keep_prob)
        
        #w2 = weight([2000,2000])
        #b2 = bias([2000])
        w2 = tf.get_variable('w2',shape=[800,800],initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable('b2',shape=[800],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w2)+b2)
        out = tf.nn.dropout(out,self.keep_prob)
        
        #w3 = weight([2000,10])
        #b3 = bias([10])
        w3 = tf.get_variable('w3',shape=[800,10],initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable('b3',shape=[10],initializer=tf.zeros_initializer())
        self.out = tf.matmul(out,w3)+b3

        self.params = [w1,b1,w2,b2,w3,b3]
        self.anchor = [tf.zeros(p.shape) for p in self.params]
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=self.out))
        self.ewc_loss = tf.constant(0.0,shape=[])
        #self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr,epsilon=self.eps).minimize(self.loss)
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        
        prob_list = tf.nn.softmax(self.out)
        class_index = tf.to_int32(tf.multinomial(tf.log(prob_list),1)[0][0])
        self.sample_loss = tf.log(prob_list[0,class_index])
        self.sample_grad = tf.gradients(self.sample_loss,self.params)

        #self.sess.run(tf.global_variables_initializer())

    def augment_loss_ewc(self,x,sample_count=250):
        #Save good params
        anchor = [np.array(self.sess.run(v),copy=True) for v in self.params]
        #Find Fisher matrices
        self.fisher = [np.zeros(v.get_shape().as_list()) for v in self.params]
        M = len(self.params)
        #sample_count = 200
        
        #prob_list = tf.nn.softmax(self.out)
        #class_index = tf.to_int32(tf.multinomial(tf.log(prob_list),1)[0][0])
        #sample_loss = tf.log(prob_list[0,class_index])
        
        #t = current_milli_time()
        sample_ind = np.random.randint(0,x.shape[0],sample_count)
        for i in range(sample_count):
            #sample = x[np.random.randint(x.shape[0])]
            sample = x[sample_ind[i]]
            #pred = self.test(np.array([sample]))
            #o,s,c,grad = self.sess.run([self.out,prob_list,class_index,tf.gradients(sample_loss,self.params)],feed_dict={self.x:np.array([sample]),self.keep_prob:1})
            #grad = self.sess.run(tf.gradients(self.sample_loss,self.params),feed_dict={self.x:np.array([sample]),self.keep_prob:1,self.learning_rate:self.lr})
            grad = self.sess.run(self.sample_grad,feed_dict={self.x:np.array([sample]),self.keep_prob:1,self.learning_rate:self.lr})
            #self.sess.run(sample_grad)
            for v in range(len(self.params)):
                #print np.max(grad[v]),np.sum(grad[v]>0),np.prod(grad[v].shape)
                self.fisher[v] += np.square(grad[v])#/float(sample_count)
                print np.max(grad[v])
        #print current_milli_time()-t
        for v in range(M):
            self.fisher[v] = self.fisher[v]/float(sample_count)
        #    #print 'maxval in fisher:',np.max(self.fisher[v])
        #Compute the EWC loss
        #self.ewc_loss = self.loss
        self.ewc_loss = tf.constant(0.0,shape=[])
        #for v in range(M):
        #    #self.ewc_loss += tf.reduce_sum(tf.multiply(np.float32(self.fisher[v]),tf.square(self.params[v]-anchor[v])))
        #    #self.ewc_loss = tf.add(self.ewc_loss,tf.reduce_sum(tf.multiply(np.float32(self.fisher[v]),tf.square(self.params[v]-anchor[v]))))
        #    #ewc_loss += (1/self.learning_rate)*tf.reduce_sum(tf.multiply(np.float32(self.fisher[v]),tf.square(self.params[v]-anchor[v])))
        #    #self.ewc_loss += tf.reduce_sum(tf.multiply(np.float32(self.fisher[v]),tf.square(self.params[v]-anchor[v])))
        #Change train step in graph to reduce ewc_loss
        self.ewc_loss = tf.reduce_sum([tf.reduce_sum(tf.multiply(np.float32(self.fisher[v]),tf.square(self.params[v]-anchor[v]))) for v in range(M)])
        self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize((self.lmbda/2)*self.ewc_loss+self.loss)
        #self.train_step = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss+self.ewc_lambda*self.ewc_loss)
        #self.sess.run(tf.global_variables_initializer())
        del anchor
        #self.lmbda -= 100

    def simple_hyperopt(self,x,y,vx,vy,use_ewc=False):
        #just train and revert
        saver = tf.train.Saver()
        tmp_file = 'tmp.ckpt'
        saver.save(self.sess,tmp_file)
        ind = range(x.shape[0])
        best = {'bz':100,'lr':1e-2,'lmbda':10,'epch':600,'val':0,'loss':999}
        #lr_beta = [1,1]
        #epch_beta = [1,1]
        #bz_beta = [1,1]
        lr_range = [5e-7,1e-2]
        r = 25
        while r > 0:
            #t = current_milli_time()
            saver.restore(self.sess,tmp_file)
            #bz = np.random.choice([50,150,200])+np.random.randint(50)
            e = np.random.random()
            if e < 0.5:
                lr = best['lr']
            else:
                #lr = a+(b-a)*np.random.random()
                lr = np.random.choice(np.arange(lr_range[0],lr_range[1],(lr_range[1]-lr_range[0])/10000.))
                #lr = a+(b-a)*(np.random.beta(*lr_beta))
            e = np.random.random()
            if e < 0.5:
                epch = best['epch']
            else:
                #epch = np.random.randint(100,1000)
                epch = int(100+(1000-100)*np.random.random()) #int(100+(2000-100)*np.random.beta(*epch_beta))
            e = np.random.random()
            if e < 0.5:
                bz = best['bz']
            else:
                bz = np.random.choice(range(20,250)) #int(20+(250-20)*np.random.beta(*bz_beta))
            for _ in range(epch):
                bi = np.random.choice(ind,bz) #batch index
                _,loss,eloss = self.sess.run([self.train_step,self.loss,self.ewc_loss],feed_dict={self.x:x[bi],self.y:y[bi],
                                                                   self.keep_prob:(1-self.dropout),self.learning_rate:lr})
                if np.isnan(loss):
                    #lr_beta[1] += 1
                    lr_range[1] = lr
                    #while np.isnan(loss):
                    #    lr_range[1] /= 3.
                    #    print lr_range[1]
                    #    loss = self.sess.run(self.loss,feed_dict={self.x:x[bi],self.y:y[bi],
                    #                                               self.keep_prob:(1-self.dropout),self.learning_rate:lr_range[1]})
                    #epch_beta[1] += 1
                    break
                #else:
                #    lr_beta[0] += 1
                #print '>>>',loss
            #for _ in range(epch):
            #    self.sess.run(self.train_step,feed_dict={self.x:x[bi],self.y:y[bi],self.keep_prob:(1-self.dropout)})
            pvy = self.sess.run(self.out,feed_dict={self.x:vx,self.keep_prob:1})
            val = acc(pvy,vy)
            print lr,bz,epch,val,loss,self.lmbda/2.,eloss
            if val >  best['val']:
                best['val'] = val
                best['loss'] = loss
                best['bz'] = bz
                best['lr'] = lr
                best['epch'] = epch
                #best['lmbda'] = lmbda
                #lr_beta[0] += 1
                #bz_beta[0] += 1
                #epch_beta[1] += 1
            #else:
            #    #lr_beta[1] += 1/3.
            #    #bz_beta[1] += 1/3.
            #    #epch_beta[0] += 1/3.
            #print lr_beta
            r -= 1
            if r == 0:
                if np.isnan(best['loss']):
                    r = 40
                    continue
            #print current_milli_time()-t
        print 'Hyperparam config:',best
        self.lr = best['lr']
        self.bz = best['bz']
        self.epch = best['epch']
        #self.lmbda = best['lmbda']
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
        self.simple_hyperopt(x,y,vx,vy,use_ewc=use_ewc)
        if log == True:
            print 'Training...'
        #if use_ewc == True:
        #    self.augment_loss_ewc(x)
        for epoch in range(self.epch):
            bi = np.random.choice(ind,self.bz) #batch index
            _,loss,py = self.sess.run([self.train_step,self.loss,self.out],feed_dict={self.x:x[bi],
                                        self.y:y[bi],self.keep_prob:(1-self.dropout),self.learning_rate:self.lr})
            if epoch%100 == 0:
                pvy = self.sess.run(self.out,feed_dict={self.x:vx,self.keep_prob:1})
                if log == True:
                    print 'Epoch',str(epoch)+': \tloss '+str(loss)+' \tval acc '+str(acc(pvy,vy))
        
        if use_ewc == True:
            self.augment_loss_ewc(x)
        
        #self.sess.close()

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
