import numpy as np
import time

batch_size = 2
input_dim = 2
outer_len = 3
inner_len = 3

def tensorflow_test():
    import tensorflow as tf
    nested_input = tf.placeholder(tf.float32, shape=[outer_len, inner_len, input_dim])

    variable = tf.Variable(np.float32(1.0))

    def inner_func(curr, prev):
        return curr + prev# + variable

    def outer_func(curr, prev):
        inner_res = tf.scan(
                fn=inner_func,
                elems=curr,
                initializer=tf.zeros([input_dim])
            )
        return prev + inner_res

    # nested_input.set_shape
    outputs = tf.scan(
            fn=outer_func,
            elems=nested_input,
            initializer=tf.zeros([inner_len, input_dim])
        )

    loss = tf.reduce_sum(outputs)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    # train_op = optimizer.minimize(loss)
    grad = tf.gradients(loss, [variable])

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init_op)
    #    nested_input_np = np.ones([outer_len, inner_len, input_dim]).astype(np.float32)
    #     feed_dict = {nested_input: nested_input_np}
    #     # print '=' * 50
    #     # print 'inputs:'
    #     # for inp in nested_input_np:
    #     #     print '-' * 50
    #     #     print inp
    #     outer_res = sess.run(outputs, feed_dict=feed_dict)
    #     print '=' * 50
    #     print 'outputs:'
    #     for res in outer_res:
    #         print '-' * 50
    #         print res
    #     # print '=' * 50
    #     # print 'gradient:'
    #     # print sess.run(grad, feed_dict=feed_dict)

def theano_test():
    import theano
    import theano.tensor as T
    theano.config.floatX = 'float32'

    floatX = theano.config.floatX

    tensor4 = T.TensorType(dtype=floatX, broadcastable=(False, False, False, False))
    
    nested_input = tensor4('nested_input')

    def np_rand(shape):
        return np.random.rand(*shape).astype(np.float32)

    W_inner = theano.shared(np_rand([input_dim, input_dim]), 'W_inner')
    W_outer = theano.shared(np_rand([input_dim, input_dim]), 'W_outer')

    def inner_func(curr, prev):
        return curr + T.dot(prev, W_inner)

    def outer_func(curr, prev):
        inner_res, inner_updates = theano.scan(
                fn=inner_func,
                sequences=curr,
                outputs_info=prev
            )

        return T.dot(prev, W_outer) + inner_res[-1]

    outputs, outer_updates = theano.scan(
            fn=outer_func,
            sequences=nested_input,
            outputs_info=T.zeros([batch_size, input_dim])
        )

    loss = T.sum(outputs)
    grads = theano.grad(loss, [W_inner, W_outer])

    func = theano.function([nested_input], grads, updates=outer_updates)

    nested_input_np = np.ones([outer_len, inner_len, batch_size, input_dim]).astype(np.float32)
    beg_time = time.time()
    func(nested_input_np)
    print time.time() - beg_time

# theano_test()
tensorflow_test()
