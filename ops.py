import tensorflow as tf


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay = self.momentum,
                                            updates_collections = None,
                                            epsilon = self.epsilon,
                                            scale = True,
                                            scope = self.name)


# Linear
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False, name="bob__"):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(name + str("Matrix"), [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(name + str("bias"), [output_size],
            initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix) + bias

# Conv2D Layer
def conv2d(input_, out_channels, filter_h=5, filter_w=5, stride_vert=2, stride_horiz=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        # Get the number of input channels
        in_channels = input_.get_shape()[-1]

        # Construct filter
        w = tf.get_variable('w', [filter_h, filter_w, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_, w, strides=[1, stride_vert, stride_horiz, 1], padding='SAME')

        # Add bias
        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

# Deconv2D Layer
def deconv2d(value, output_shape, filter_h=5, filter_w=5, stride_vert=2, stride_horiz=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        # Get the number of input/output channels
        in_channels = value.get_shape()[-1]
        out_channels = output_shape[-1]

        # Construct filter
        w = tf.get_variable('w', [filter_h, filter_w, out_channels, in_channels],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(value, w, output_shape=output_shape,
                                        strides=[1, stride_vert, stride_horiz, 1])

        # Add bias
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv

# Leaky RELU
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
