import tensorflow as tf
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
l2_regularizer= tf.contrib.layers.l2_regularizer(1.0)
xavier = tf.contrib.layers.xavier_initializer_conv2d() 
def infer(input):
    with tf.variable_scope('conv1_'):
        network = first_conv(input, 64, name = 'conv1')
        network = block(network, 'conv1_23', 64)
    with tf.variable_scope('conv2_'):
        network = first_conv(network, 128, name = 'conv2')
        network = block(network, 'conv2_23', 128)
        network = block(network, 'conv2_45', 128)
    with tf.variable_scope('conv3_'):
        network = first_conv(network, 256, name = 'conv3')
        network = block(network, 'conv3_23', 256)
        network = block(network, 'conv3_45', 256)
        network = block(network, 'conv3_67', 256)
        network = block(network, 'conv3_89', 256)
    with tf.variable_scope('conv4_'):
        network = first_conv(network, 512, name = 'conv4')
        network = block(network, 'conv4_23', 512)
    with tf.variable_scope('feature'):
        BATCH_SIZE = network.get_shape()[0]
        feature = tf.layers.dense(tf.reshape(network,[BATCH_SIZE, -1]), 512)
    return feature


def prelu(x, name = 'prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', x.get_shape()[-1], initializer=tf.constant_initializer(0.25), dtype = tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - abs(x)) * 0.5
    return pos + neg

def first_conv(input, num_output, name):
    
    zero_init = tf.zeros_initializer()
    network = tf.layers.conv2d(input, num_output, kernel_size = [3, 3], strides = (2, 2), padding = 'same', kernel_initializer = xavier, bias_initializer = zero_init, kernel_regularizer = l2_regularizer, bias_regularizer = l2_regularizer)
    network = prelu(network, name = name)
    return network


def block(input, name, num_output):
    with tf.variable_scope(name):
        network = tf.layers.conv2d(input, num_output, kernel_size = [3, 3], strides = [1, 1], padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01), use_bias = False , kernel_regularizer = l2_regularizer)
        network = prelu(network, name = 'name'+ '1')
        network = tf.layers.conv2d(network, num_output, kernel_size = [3, 3], strides = [1, 1], padding = 'same', kernel_initializer = tf.random_normal_initializer(stddev=0.01), use_bias = False, kernel_regularizer = l2_regularizer)
        network = prelu(network, name = 'name'+ '2')
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

