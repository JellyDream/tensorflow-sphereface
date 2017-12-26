import tensorflow as tf
import numpy as np
import random
weight = tf.reshape(tf.constant(np.arange(16), dtype = tf.float32), [4,4])
norm_weight = tf.nn.l2_normalize(weight, dim = 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    norm_out, weight_out = sess.run([norm_weight, weight])
    print(norm_out, weight_out)
