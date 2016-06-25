import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

OUTER_LOOP_MAX = 5
INNER_LOOP_MAX = 3

inner = tf.placeholder("float32", [1, INNER_LOOP_MAX])
outer = tf.placeholder("float32", [1, OUTER_LOOP_MAX])

X = tf.Variable(np.float32(1.0))

threshold = tf.constant(1)

def outer_cond_func(outer_counter, outer_accum, outer_array):
    return tf.less(outer_counter, OUTER_LOOP_MAX)

def outer_body_func(outer_counter, outer_accum, outer_array):
    # REF: tf.concat(concat_dim, values, name='concat')
    outer_concat = tf.concat(0, [[0], tf.expand_dims(outer_counter, 0)])
    # REF: tf.slice(input_, begin, size, name=None)
    outer_slices = tf.slice(outer_array, outer_concat, [1, 1])

    outer_num = tf.reduce_sum(outer_slices)

    def inner_cond_func(inner_counter, inner_accum, inner_array):
        return tf.less(inner_counter, INNER_LOOP_MAX)

    def inner_body_func(inner_counter, inner_accum, inner_array):
        inner_concat = tf.concat(0, [[0], tf.expand_dims(inner_counter, 0)])
        inner_slices = tf.slice(inner_array, inner_concat, [1, 1])
        inner_num = tf.reduce_sum(inner_slices)
        inner_accum += inner_num * outer_num * X

        inner_counter += 1
        return inner_counter, inner_accum, inner_array

    _, inside_summed_products, _ = control_flow_ops.while_loop(
        cond=inner_cond_func,
        body=inner_body_func,
        loop_vars=[tf.constant(0), tf.constant(0.0), inner]
    )

    def true_func():
        return 2*outer_num

    def false_func():
        return 3*outer_num

    # cond_num = control_flow_ops.cond(tf.less(outer_counter, threshold), true_func, false_func)
    outer_accum = tf.add(outer_accum, inside_summed_products)
    # outer_accum = tf.add(outer_accum, cond_num)
    outer_counter += 1
    return outer_counter, outer_accum, outer_array

_, value, _ = control_flow_ops.while_loop(
    cond=outer_cond_func,
    body=outer_body_func,
    loop_vars=[tf.constant(0), tf.constant(0.0), outer] # outer_counter, outer_accum, outer_array
)
# control_flow_ops.switch()
# print value

loss = value * X

grads = tf.gradients(loss, [X])
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in xrange(1):
    feed_dict = {inner: [[1.0, 2.0, 3.0]], outer: [[4.0, 5.0, 6.0, 7.0, 8.0]]}
    print sess.run([train_op, loss], feed_dict=feed_dict)
    print sess.run(grads, feed_dict=feed_dict)
  # print sess.run([value, loss], feed_dict=feed_dict)
