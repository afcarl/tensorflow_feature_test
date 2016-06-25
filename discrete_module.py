import numpy as np
import tensorflow as tf

b_size = 16
i_size = 32
o_size = 8

np.random.seed(3)
tf.set_random_seed(3)

x = tf.placeholder(tf.float32, shape=[b_size, i_size])

W_a = tf.Variable(tf.random_uniform([i_size, 1]), name='W_a')
b_a = tf.Variable(tf.zeros([1]), name='b_a')
p_a = tf.nn.sigmoid(tf.matmul(x, W_a) + b_a)

W_1 = tf.Variable(tf.random_uniform([i_size, o_size]), name='W_1')
b_1 = tf.Variable(tf.zeros([o_size]), name='b_1')
p_1 = tf.nn.softmax(tf.matmul(x, W_1) + b_1)

W_2 = tf.Variable(tf.random_uniform([i_size, o_size]), name='W_2')
b_2 = tf.Variable(tf.zeros([o_size]), name='b_2')
p_2 = tf.nn.softmax(tf.matmul(x, W_2) + b_2)

# b_y = tf.Variable(tf.zeros([o_size]), name='b_y')

# y = p_a * p_1 + (1 - p_a) * p_2

# act = tf.stop_gradient(tf.cast(tf.greater(p_a, tf.constant(0.5, name='0.5')), tf.float32))
act = tf.cast(tf.greater(p_a, tf.constant(0.5, name='0.5')), tf.float32)

y = (act * p_1 + (1 - act) * p_2) * p_a
# y = (act * p_1 + (1 - act) * p_2 + b_y) * p_a

t = tf.placeholder(tf.float32, shape=[b_size, o_size])
l = tf.reduce_mean(tf.square(y - t))

optimizer = tf.train.AdamOptimizer(0.1)
grads_and_vars = optimizer.compute_gradients(l)
train_op = optimizer.apply_gradients(grads_and_vars)

# mannually check each grad & var
# grads = []
# names = []

# for g, v in tuple(grads_and_vars):
# 	if g is not None:
# 		grads.append(g)
# 	else:
# 		grads.append(tf.constant('None'))
# 	names.append(v.name)

# train_op = optimizer.minimize(l)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init_op)

	summary_writer = tf.train.SummaryWriter('./', sess.graph)
	summary_writer.flush()

	x_np = np.random.rand(b_size, i_size).astype(np.float32)
	t_np = np.random.rand(b_size, o_size).astype(np.float32)

	feed_dict = {
		x : x_np,
		t : t_np
	}

	# print sess.run(W_a).reshape(1, -1)
	# grads_np = sess.run(grads, feed_dict=feed_dict)
	# for grad, name in zip(grads_np, names):
	# 	print name, grad if isinstance(grad, str) else grad.shape
	# sess.run(train_op, feed_dict=feed_dict)
	# print sess.run(W_a).reshape(1, -1)

	grads_np = sess.run(train_op, feed_dict=feed_dict)