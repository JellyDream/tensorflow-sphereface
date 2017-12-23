import tensorflow as tf
import argparse
import cv2
import random
import math
import os
import numpy as np
import time
from tensorflow.python.framework import ops
marginInnerProduct_module = tf.load_op_library('../marginInnerProduct/libtf_marginInnerProduct.so')

@ops.RegisterGradient('MarginInnerProduct')
def _margin_inner_product_grad(op,top_diff):
	input=op.inputs[0]
	weight=op.inputs[1]
	m_value=op.inputs[2]
	lambda_value=op.inputs[3]
        label = op.inputs[4]
        grad_input, grad_weight = marginInnerProduct_module.margin_inner_product_grad(input, weight, m_value, lambda_value, label, top_diff)
	return [grad_input, grad_weight, None, None, None]


data_root_dir = 'data/'
data_list = 'webface_list.txt'

BATCH_SIZE = 256
M_VALUE = 4

basic_learning_rate = 0.1
weight_decay = 0.0005
step_value = [16000, 20000, 24000]
max_iter = 28000
factor = 0.1

lambda_base = 1000
lambda_gamma = 0.12
lambda_power = 1
lambda_min = 5
begin_iteration = 0

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

l2_regularizer= tf.contrib.layers.l2_regularizer(1.0)
xavier = tf.contrib.layers.xavier_initializer_conv2d() 

def infer(input):
    with tf.variable_scope('conv1_'):
        network = first_conv(input, 64)
        network = prelu(network)
        network = block(network, 'conv1_23', 64)
    with tf.variable_scope('conv2_'):
        network = first_conv(network, 128)
        network = block(network, 'conv2_23', 128)
        network = block(network, 'conv2_45', 128)
    with tf.variable_scope('conv3_'):
        network = first_conv(network, 256)
        network = block(network, 'conv3_23', 256)
        network = block(network, 'conv3_45', 256)
        network = block(network, 'conv3_67', 256)
        network = block(network, 'conv3_89', 256)
    with tf.variable_scope('conv4_'):
        network = first_conv(network, 512)
        network = block(network, 'conv4_23', 512)
    with tf.variable_scope('feature'):
        BATCH_SIZE = network.get_shape()[0]
        feature = tf.layers.dense(tf.reshape(network,[BATCH_SIZE, -1]), 512)
    return feature


def prelu(x):
    t = tf.Variable(0.25, tf.float32)
    return tf.maximum(x, tf.multiply(x, t))

def first_conv(input, num_output):
    
    zero_init = tf.zeros_initializer()
    network = tf.layers.conv2d(input, num_output, kernel_size = [3, 3], strides = (2, 2), padding = 'same', kernel_initializer = xavier, bias_initializer = zero_init, kernel_regularizer = l2_regularizer, bias_regularizer = l2_regularizer)
    network = prelu(network)
    return network


def block(input, name, num_output):
    with tf.variable_scope(name):
        network = tf.layers.conv2d(input, num_output, kernel_size = [3, 3], strides = [1, 1], padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01), use_bias = False , kernel_regularizer = l2_regularizer)
        network = prelu(network)
        network = tf.layers.conv2d(network, num_output, kernel_size = [3, 3], strides = [1, 1], padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01), use_bias = False, kernel_regularizer = l2_regularizer)
        network = prelu(network)
        network = tf.add(input, network)
        return network

def get_normal_loss(input, label, num_output, lambda_value, m_value = 4):
    feature_dim = input.get_shape()[1]
    weight = tf.get_variable('weight', shape = [num_output, feature_dim], regularizer = l2_regularizer, initializer = xavier)
    prob_distribution = tf.one_hot(label, num_output)
    weight = tf.nn.l2_normalize(weight, dim = 1)
    label_float = tf.cast(label, tf.float32)
    margin_out = marginInnerProduct_module.margin_inner_product(input, weight, tf.constant(m_value), lambda_value, label_float)
    #margin_out = tf.layers.dense(input, num_output)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=margin_out, labels = prob_distribution))
    return loss

def file_list(data_dir, list_txt):
    filenames = []
    with open(list_txt, 'rt') as f:
        for line in f:
            line = line.rstrip()
            fn = os.path.join(data_dir, line)
            filenames.append(fn)
    return filenames

used_count = -1
def load_data(filenames, batch_size, do_shuffle):
    global used_count
    if used_count == -1:
       if do_shuffle:
            random.shuffle(filenames)
       used_count = 0
    if used_count > len(filenames) - BATCH_SIZE:
        used_count = 0
        if do_shuffle:
            random.shuffle(filenames)

    batch_data = np.zeros((batch_size, 112, 96, 3))
    batch_label = np.zeros((batch_size), dtype = np.int32)
    for i in range(BATCH_SIZE):
        img_name = filenames[used_count + i].split(' ')[0]
        label = int(filenames[used_count + i].split(' ')[1])
        img = cv2.imread(img_name)
        batch_data[i,:] = (img-127.5)/128.0
        batch_label[i] = label
    used_count = used_count + BATCH_SIZE
    return batch_data, batch_label

def get_multistep_lr(iter_):
    return basic_learning_rate * math.pow(factor, sum([1 for value in step_value if value < iter_]))

def get_lambda_value(iter_):
    return lambda_base * math.pow( lambda_gamma * (iter_ -begin_iteration) + 1, -lambda_power)


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
            print(used_count)
            #print(i, lambda_value_out, get_multistep_lr(iter_))
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
