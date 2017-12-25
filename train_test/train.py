import tensorflow as tf
import argparse
import cv2
import random
import os
import numpy as np
import time

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

def train():
    learning_rate = tf.placeholder(tf.float32)
    lambda_value = tf.placeholder(tf.float32)
    input_images = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 112, 96, 3])
    input_labels = tf.placeholder(tf.int32, shape = [BATCH_SIZE])

    features = infer(input_images)
    normal_loss = get_normal_loss(features, input_labels, 10572, lambda_value, M_VALUE)
    tf.summary.scalar('loss', normal_loss)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = normal_loss + weight_decay * sum(reg_losses)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    
    filenames = file_list(data_root_dir, data_list)

    merged_summary_op = tf.summary.merge_all()
    SAVER = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        file_writer = tf.summary.FileWriter('log', sess.graph)
        iter_ = 0
        for i in xrange(max_iter):
            begin_time = time.time()
            batch_data, batch_label = load_data(filenames, BATCH_SIZE, True)
            lambda_value_out = get_lambda_value(iter_)
            feed_dict = {learning_rate:get_multistep_lr(iter_), lambda_value: lambda_value_out, input_images:batch_data, input_labels:batch_label}
            _, loss_value,summary_str = sess.run([optimizer, normal_loss, merged_summary_op], feed_dict = feed_dict)
            file_writer.add_summary(summary_str, iter_)
            iter_ = iter_ + 1
            print(iter_, lambda_value_out, get_multistep_lr(iter_))
            print('%d:iteration : %f iter/s, loss,:%f'%(i, 1/(time.time()-begin_time), loss_value))
            if (i+1)%2000 == 0:
                save_path = SAVER.save(sess, "./ckpt/" + str(i+1) + "model.ckpt")
                print "Model saved in file: ", save_path

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

def get_features(model_ckpt):
    input_images = tf.placeholder(tf.float32, shape = [1, 112, 96, 3])
    features = infer(input_images)
    filenames = file_list('/home/scw4750/github/sphereface_tensorflow_a-softmax/train/data', '/home/scw4750/github/sphereface_tensorflow_a-softmax/test/test_list.txt')
    SAVER = tf.train.Saver()
    with tf.Session() as sess:
         SAVER.restore(sess, model_ckpt)
         batch_data = load_test_data(filenames, True)
         data_len = batch_data.shape[0]
         all_features = np.zeros([data_len, features.get_shape()[1]])

         for i in range(data_len):
             img = np.zeros([1,112,96,3])
             img[:] = batch_data[i,:]
             feature = sess.run([features], feed_dict = {input_images:img})[0]
             all_features[i,:] = feature
    return all_features

if __name__ == '__main__':
    train()
    #all_features = get_features("./ckpt/model.ckpt")
    #print(all_features)
