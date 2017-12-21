import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.python.framework import ops
marginInnerProduct_module = tf.load_op_library('./libtf_marginInnerProduct.so')

@ops.RegisterGradient('MarginInnerProduct')
def _margin_inner_product_grad(op,top_diff):
	input=op.inputs[0]
	weight=op.inputs[1]
	m_value=op.inputs[2]
	lambda_value=op.inputs[3]
        label = op.inputs[4]
        grad_input, grad_weight = marginInnerProduct_module.margin_inner_product_grad(input, weight, m_value, lambda_value, label, top_diff)
	return [grad_input, grad_weight, None, None, None]

if __name__ == '__main__':
    import numpy as np
    import random
    import time
    from tensorflow.python.ops.gradient_checker import compute_gradient
    input = np.random.randn(32,64).astype('float32')
    input_tensor = tf.constant(input)
    weight = np.random.randn(128, 64).astype('float32')
    weight_tensor = tf.nn.l2_normalize(tf.Variable(weight), dim=1)
    m_value = tf.constant([4], dtype = tf.int32)
    lambda_value = tf.constant([5], dtype = tf.float32)
    label = tf.constant(np.arange(32), dtype = tf.int32)
    label_float = tf.cast(label, tf.float32)
    prob_distribution = tf.one_hot(label, 128)
    margin_out = marginInnerProduct_module.margin_inner_product(input_tensor, weight_tensor, m_value, lambda_value, label_float)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=margin_out, labels = prob_distribution))
    learning_rate = tf.placeholder(tf.float32)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(10000):
            print(i)
            #grads=compute_gradient([inp1,inp2],[(16,32,3),(16,32,3)],loss,(1,),[xyz1,xyz2])
            trainloss,_, weight_value, input_value = sess.run([loss, optimizer, weight_tensor, input_tensor], feed_dict = {learning_rate:0.1})
            print(trainloss, weight_value.mean(), input_value.mean())

