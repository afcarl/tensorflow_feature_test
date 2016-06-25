import numpy as np
import tensorflow as tf

b_size = 4
i_size = 5
h_size = 4
seq_len = 10

weights_hh = tf.Variable(np.random.rand(h_size, h_size))
weights_ih = tf.Variable(np.random.rand(i_size, h_size))

initial_h  = tf.constant(np.zeros((b_size, h_size)))
sequence_i = tf.placeholder(tf.float64, shape=[None, b_size, i_size])

def step_func(h, i):
	hh = tf.matmul(h, weights_hh)
	ih = tf.matmul(i, weights_ih)
	return tf.nn.tanh(hh + ih)

out = tf.scan(step_func, sequence_i, initializer=initial_h)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(out, feed_dict={sequence_i: np.random.rand(seq_len, b_size, i_size)}).shape)

