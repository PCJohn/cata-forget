import cv2
import numpy as np
import random
import tensorflow as tf

import mnist
import tasks

def weight(shape):
    intial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(intial)

def bias(shape):
    intial = tf.constant(0.1,shape=shape)
    return tf.Variable(intial)

def start_sess():
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    return sess

def acc(py,y):
    return np.mean(py.argmax(axis=1)==y.argmax(axis=1))

class FullyConn():
    def __init__(self,sess):
        self.lr = 1e-5
        self.eps = 1e-8
        self.bz = 100
        self.epch = 500
        self.dropout = 0.5 #Actually 1-dropout

        self.sess = sess
        self.x = tf.placeholder(tf.float32,shape=[None,1,28,28])
        self.y = tf.placeholder(tf.float32,shape=[None,10])
        self.keep_prob = tf.placeholder(tf.float32)
        
        d = 28*28*1
        out = tf.reshape(self.x,[-1,d])
        
        w1 = weight([d,2000])
        b1 = bias([2000])
        #w1 = tf.get_variable('w1',shape=[d,4000],initializer=tf.contrib.layers.xavier_initializer())
        #b1 = tf.get_variable('b1',shape=[4000],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w1)+b1)
        out = tf.nn.dropout(out,self.keep_prob)
        
        w2 = weight([2000,2000])
        b2 = bias([2000])
        #w2 = tf.get_variable('w2',shape=[4000,4000],initializer=tf.contrib.layers.xavier_initializer())
        #b2 = tf.get_variable('b2',shape=[4000],initializer=tf.zeros_initializer())
        out = tf.nn.relu(tf.matmul(out,w2)+b2)
        out = tf.nn.dropout(out,self.keep_prob)
        
        w3 = weight([2000,10])
        b3 = bias([10])
        #w3 = tf.get_variable('w3',shape=[4000,10],initializer=tf.contrib.layers.xavier_initializer())
        #b3 = tf.get_variable('b3',shape=[10],initializer=tf.zeros_initializer())
        self.out = tf.matmul(out,w3)+b3

        self.params = [w1,b1,w2,b2,w3,b3]
        self.anchor = [tf.zeros(p.shape) for p in self.params]
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=self.out))
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr,epsilon=self.eps).minimize(self.loss)
        
        #self.ewc_loss = self.loss+self.params

        self.sess.run(tf.global_variables_initializer())

    def init_loss(self):
        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.y,logits=self.out))
    
    def ecw_loss(self,ewc):
        self.loss += (ewc/2)*tf.reduce_sum(tf.multiply())

    def train(self,train,val=None,ewc=0):
        x,y = train
        if val != None:
            vx,vy = val
        ind = range(x.shape[0])
        print 'Training...'
        #self.init_loss()#Base: crossentropy
        #F = self.fisher(x)    #Fisher
        for epoch in range(self.epch):
            bi = np.random.choice(ind,self.bz) #batch index
            #print '>>>',y[bi].shape
            _,loss,py = self.sess.run([self.train_step,self.loss,self.out],feed_dict={self.x:x[bi],
                                        self.y:y[bi],self.keep_prob:(1-self.dropout)})
            if epoch%100 == 0:
                pvy = self.sess.run(self.out,feed_dict={self.x:vx,self.keep_prob:1})
                print 'Epoch',str(epoch)+': \tloss '+str(loss)+' \tval acc '+str(acc(pvy,vy))

    def test(self,x,y=None):
        py = self.sess.run(self.out,feed_dict={self.x:x,self.keep_prob:1})
        if y is None:
            return py
        else:
            return (py,acc(py,y))

    def fisher(self,x):
        F = [tf.zeros(p.get_shape().as_list()) for p in self.params]
        for i in range(2000):
            k = np.random.choice(range(x.shape[0]))
            dt = self.sess.run(tf.gradients(tf.log(tf.nn.softmax(self.out)[0,k]),self.anchor_params),
                                    feed_dict={self.x:x[k:k+1]})#,self.keep_prob:1})
            for j in range(len(F)):
                F[j] += np.square(dt[j])
        return F

#(ds,vds),(pds_list,pvds_list) = mnist_permute.load(permute=1)
ds,vds = mnist.load()
x,y = map(np.float32,zip(*ds))
vx,vy = map(np.float32,zip(*vds))
print x.shape,y.shape,'--',vx.shape,vy.shape

#EXPERIMENT
sess = start_sess()
model = FullyConn(sess)
print 'Trask 0: simple MNIST...'
model.train((x,y),val=(vx,vy))

#Permutation tasks
val_ds = [(vx,vy)]
for i in range(10):
    tds,tvds = tasks.permute(ds,vds)
    tx,ty = map(np.float32,zip(*tds))
    tvx,tvy = map(np.float32,zip(*tvds))
    val_ds.append((tvx,tvy))

    print 'Training on task '+str(i+1)+'...'
    model.epch += 100
    model.train((tx,ty),val=(tvx,tvy),ewc=0)
    for j,(ovx,ovy) in enumerate(val_ds):
        _,vacc = model.test(ovx,y=ovy)
        print 'Test on task '+str(j)+' '+str(vacc)

#TODO: Split tasks

sess.close()
