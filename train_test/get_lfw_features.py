import tensorflow as tf
import argparse
import cv2
import random
import os
import numpy as np
import time
import h5py
from network import *
from utils import *

#The following data augmentation will be completed as soon as possible
#do_mirror = True
#Due to the vibration of face landmarks, the shift, rotation and zoom will increase the robustness of feature representation
#do_shift = true  
#do_zoom = true
#do_rotation = true

#To increase the robustness
#do_modHSV = true
#do_modRGB = true
#do_smooth = true
#do_jpgCompression = true


def load_test_data(filenames, with_label):
    files_len = len(filenames)
    batch_data = np.zeros([files_len, 112, 96, 3])
    for i in range(files_len):
        file_name = filenames[i].rstrip('\n')
        if with_label:
            img_name = file_name.split(' ')[0]
        else:
            img_name = file_name
        img = cv2.imread(img_name)
        batch_data[i,:] = (img - 127.5)/128.0
    return batch_data

def get_features(model_ckpt, filenames):
    input_images = tf.placeholder(tf.float32, shape = [1, 112, 96, 3])
    features = infer(input_images)
    SAVER = tf.train.Saver()
    with tf.Session() as sess:
         SAVER.restore(sess, model_ckpt)
         batch_data = load_test_data(filenames, False)
         data_len = batch_data.shape[0]
         all_features = np.zeros([data_len, features.get_shape()[1]])

         for i in range(data_len):
             print(i)
             img = np.zeros([1,112,96,3])
             img[:] = batch_data[i,:]
             feature = sess.run([features], feed_dict = {input_images:img})[0]
             #feature_ = sess.run([features], feed_dict = {input_images:img[:,::-1,:]})[0]
             #all_features[i,:] = np.concatenate([feature, feature_], axis=1)
             all_features[i,:] = feature
    return all_features

if __name__ == '__main__':
    filenames = file_list(test_data_dir, test_list)
    all_features = get_features(test_ckpt_model, filenames)
    file=h5py.File('lfw_features.h5')
    file.create_dataset('name',data=filenames)
    file.create_dataset('feature',data=all_features)
    file.close()
