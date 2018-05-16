#####################################################################################################
# testing VGG face model using a pre-trained model
# written by Zhifei Zhang, Aug., 2016
#####################################################################################################

import os
from vgg_face import vgg_face
from scipy.misc import imread, imresize
import tensorflow as tf
import numpy as np
import gc
from matplotlib import pyplot as plt

path_to_dataset_folder = '/home/prithvi/dsets/MSCeleb/MSCeleb/'
path_to_save_features = '/home/prithvi/dsets/MSCeleb/features.npy'

model_path = './Tensorflow-VGG-face/vgg-face.mat'

class FeatureExtractor():
    def __init__(self):
        # build the graph
        print 'Initializing feature extractor'
        print 'Loading model from '+model_path
        graph = tf.Graph()
        with graph.as_default():
            input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
            output, average_image, class_names = vgg_face(model_path, input_maps)
            feats = output['fc7']
        self.graph = graph
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config,graph=graph)
        self.feats = feats
        self.average_image = average_image
        self.input_maps = input_maps
        self.u = 0
        print 'Feature extractor ready!'

    def preproc(self,img):
        if img.shape[0] > 250:
            self.u += 1
            print self.u
        img = imresize(img,[224,224])
        img = img-self.average_image
        return img

    def features(self,img_list):
        img_list = [self.preproc(img) for img in img_list]
        [ft] = self.sess.run([self.feats],feed_dict={self.input_maps:img_list})
        ft = ft[:,0,0,:]
        return ft
    
    def reset(self):
        self.sess.close()
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config,graph=self.graph)

    def stop(self):
        self.sess.close()
        print 'Feature extractor session closed'

#Extract features for all images in a folder
def all_feat_in_folder(base):
    subd = os.listdir(base)
    fe = FeatureExtractor()
    feat = {}
    pid = 0
    for p in subd[:]:
        print 'Extracting features for images of person number:',pid
        pth = os.path.join(base,p)
        pth_list = []
        img_list = []
        for img_file in os.listdir(pth)[:]:
            img_path = os.path.join(pth,img_file)
            pth_list.append(img_path)
            img = imread(img_path,mode='RGB')
            ft = fe.features([img])
            print 'Features from person:',pid,'@ path',img_path,'of shape ==>',ft.shape#,ft[:5]
            feat.update({img_path:ft})
        pid += 1
    fe.stop()
    return feat

def extract_feat():
    all_feat = all_feat_in_folder(path_to_dataset_folder)
    np.save(path_to_save_features,all_feat)
    print 'Features saved to',path_to_save_features
