import tensorflow as tf
import argparse
import cv2
import random
import math
import os
import numpy as np

from tensorflow.python.framework import ops
marginInnerProduct_module = tf.load_op_library('/home/idealsee/github/tf_sphereface/marginInnerProduct/libtf_marginInnerProduct.so')

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
    xavier = tf.contrib.layers.xavier_initializer_conv2d() 
    zero_init = tf.zeros_initializer()
    network = tf.layers.conv2d(input, num_output, kernel_size = [3, 3], strides = (2, 2), padding = 'same', kernel_initializer = xavier, bias_initializer = zero_init, kernel_regularizer = l2_regularizer, bias_regularizer = l2_regularizer)
    return network


def block(input, name, num_output):
    with tf.variable_scope(name):
        network = tf.layers.conv2d(input, num_output, kernel_size = [3, 3], strides = [1, 1], padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01), use_bias = False , kernel_regularizer = l2_regularizer)
        network = prelu(network)
        network = tf.layers.conv2d(network, num_output, kernel_size = [3, 3], strides = [1, 1], padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01), use_bias = False, kernel_regularizer = l2_regularizer)
        network = prelu(network)
        network = tf.add(input, network)
        return network

def get_normal_loss(input, label, num_output, lambda_value, m_value):
    feature_dim = input.get_shape()[1]
    weight = tf.get_variable('weight', shape = [num_output, feature_dim], regularizer = l2_regularizer)
    weight = tf.nn.l2_normalize(weight, dim = 1)
    prob_distribution = tf.one_hot(label, num_output)
    label_float = tf.cast(label, tf.float32)
    margin_out = marginInnerProduct_module.margin_inner_product(input, weight, tf.constant(m_value), lambda_value, label_float)
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
def load_data(filenames, BATCH_SIZE, do_shuffle):
    global used_count
    if used_count == -1:
       if do_shuffle:
            random.shuffle(filenames)
       used_count = 0
    if used_count > len(filenames) - BATCH_SIZE:
        used_count = 0
        if do_shuffle:
            random.shuffle(filenames)

    batch_data = np.zeros((BATCH_SIZE, 112, 96, 3))
    batch_label = np.zeros((BATCH_SIZE))
    for i in range(used_count, used_count + BATCH_SIZE):
        img_name = filenames[i].split()[0]
        label = int(filenames[i].split()[1])
        img = cv2.imread(filenames[i])
        batch_data[i,:] = img
        batch_label[i] = label
    return batch_data, batch_label

def get_multistep_lr(iter_):
    return basic_learning_rate * math.pow(factor, sum([1 for value in step_value if value < iter_]))

def get_lambda_value(iter_):
    return lambda_base * math.pow( lambda_gamma * (iter_ -begin_iteration) + 1, -lambda_power)

if __name__ == '__main__':

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    learning_rate = tf.placeholder(tf.float32)
    lambda_value = tf.placeholder(tf.float32)
    input_images = tf.placeholder(tf.float32, shape = [BATCH_SIZE, 112, 96, 3])
    input_labels = tf.placeholder(tf.int32, shape = [BATCH_SIZE])

    features = infer(input_images)
    normal_loss = get_normal_loss(features, input_labels, 10512, lambda_value, 4)

    loss = normal_loss + weight_decay * sum(reg_losses)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss)
    
    filenames = file_list(data_root_dir, data_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        iter_ = 0
        for i in xrange(max_iter):
            batch_data, batch_label = load_data(filenames, BATCH_SIZE, True)
            lambda_value_out = get_lambda_value(iter_)
            feed_dict = {learning_rate:get_multistep_lr(iter_), lambda_value: lambda_value_out, input_images:batch_data, input_labels:batch_label}
            _, loss_value = sess.run([optimizer, normal_loss], feed_dict = feed_dict)
            iter_ = iter_ + 1
            print(i, loss_value, lambda_value_out)



