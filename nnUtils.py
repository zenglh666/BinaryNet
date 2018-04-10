import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

tf.app.flags.DEFINE_string('regularizer', 'None',
                           """regularizer for weights:L1(L1),L2(L2)""")
tf.app.flags.DEFINE_float('regularizer_norm', 0.0001,
                          """regularizer norm.""")
tf.app.flags.DEFINE_float('weight_norm_factor', 0.0001,
                          """regularizer norm.""")
tf.app.flags.DEFINE_boolean('weight_norm', False,
                           """if norm weight.""")
tf.app.flags.DEFINE_integer('bit', 1,
                               """number of bit""")
tf.app.flags.DEFINE_float('zeta', 0.0,
                          """zeta.""")
FLAGS = tf.app.flags.FLAGS
regularizer = None
    
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)

def BinarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, name='BinarizedSpatialConvolution'):
    def b_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))

            bin_w = binarize(w)
            bin_x = binarize(x)
            '''
            Note that we use binarized version of the input and the weights. Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            out = tf.nn.conv2d(bin_x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return b_conv2d

def BinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            bin_w = binarize(w)
            out = tf.nn.conv2d(tf.clip_by_value(x,-1,1), bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def AccurateBinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            for i in range(FLAGS.bit):
                if i == 0:
                    bin_w = binarize(w)
                    alpha = tf.reduce_mean(tf.abs(w))
                    w_mul = tf.multiply(bin_w, alpha)
                    w_apr = tf.identity(w_mul)
                    w_res = tf.subtract(w, w_mul)
                else:
                    bin_w = binarize(w_res)
                    alpha = tf.reduce_mean(tf.abs(w_res))
                    w_mul = tf.multiply(bin_w, alpha)
                    w_apr = tf.add(w_apr, w_mul)
                    w_res = tf.subtract(w_res, w_mul)
            out = tf.nn.conv2d(tf.clip_by_value(x,-1,1), w_apr, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def MoreAccurateBinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            for i in range(FLAGS.bit):
                if i == 0:
                    bin_w = binarize(w)
                    alpha = tf.div(tf.reduce_mean(tf.pow(tf.abs(w),FLAGS.zeta + 1), axis=[0,1,2], keep_dims=True),
                                   tf.reduce_mean(tf.pow(tf.abs(w),FLAGS.zeta), axis=[0,1,2], keep_dims=True))
                    w_mul = tf.multiply(bin_w, alpha)
                    w_apr = tf.identity(w_mul)
                    w_res = tf.subtract(w, w_mul)
                else:
                    bin_w = binarize(w_res)
                    alpha = tf.div(tf.reduce_mean(tf.pow(tf.abs(w_res),FLAGS.zeta + 1), axis=[0,1,2], keep_dims=True),
                                   tf.reduce_mean(tf.pow(tf.abs(w_res),FLAGS.zeta), axis=[0,1,2], keep_dims=True))
                    w_mul = tf.multiply(bin_w, alpha)
                    w_apr = tf.add(w_apr, w_mul)
                    w_res = tf.subtract(w_res, w_mul)
            out = tf.nn.conv2d(tf.clip_by_value(x,-1,1), w_apr, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=True, name='SpatialConvolution'):
    def conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'),
                            regularizer=regularizer)
            if FLAGS.weight_norm:
                w_mean = tf.get_variable('w_mean', [1,1,w.shape[2],1],
                            initializer=tf.constant_initializer(0.), trainable=False)
                beta = tf.get_variable('beta', [1,1,w.shape[2],1],
                            initializer=tf.constant_initializer(0.), trainable=False)
                gamma = tf.get_variable('gamma', [1,1,w.shape[2],1],
                            initializer=tf.constant_initializer(1.), trainable=False)

        if FLAGS.weight_norm:
            with tf.variable_scope(name + '/bn', reuse=(not is_training)):
                _, x_variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
                x_variance = tf.transpose(x_variance, perm=[0, 1, 3, 2])

                ema = tf.train.ExponentialMovingAverage(decay=(1-FLAGS.weight_norm_factor))
                if is_training:
                    apply_op = ema.apply([x_variance])
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_op)
                    avg_x_variance = tf.identity(x_variance)
                else:
                    avg_x_variance = ema.average(x_variance)

                w = tf.nn.batch_normalization(w, w_mean, avg_x_variance, beta, gamma, 1e-3)

        with tf.variable_scope(name, values=[x], reuse=reuse):
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d

def Affine(nOutputPlane, bias=True, name=None):
    def affineLayer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, 'Affine', values=[x], reuse=reuse):
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane],
                                initializer=tf.variance_scaling_initializer(mode='fan_avg'),
                                regularizer=regularizer)
            output = tf.matmul(reshaped, w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return affineLayer

def BinarizedAffine(nOutputPlane, bias=True, name=None):
    def b_affineLayer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, 'Affine', values=[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            bin_x = binarize(x)
            reshaped = tf.reshape(bin_x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane],
                initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return b_affineLayer

def BinarizedWeightOnlyAffine(nOutputPlane, bias=True, name=None):
    def bwo_affineLayer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, 'Affine', values=[x], reuse=reuse):
            '''
            Note that we use binarized version of the input (bin_x) and the weights (bin_w). Since the binarized function uses STE
            we calculate the gradients using bin_x and bin_w but we update w (the full precition version).
            '''
            reshaped = tf.reshape(x, [x.get_shape().as_list()[0], -1])
            nInputPlane = reshaped.get_shape().as_list()[1]
            w = tf.get_variable('weight', [nInputPlane, nOutputPlane],
                initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            bin_w = binarize(w)
            output = tf.matmul(reshaped, bin_w)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                output = tf.nn.bias_add(output, b)
        return output
    return bwo_affineLayer

def Linear(nInputPlane, nOutputPlane):
    return Affine(nInputPlane, nOutputPlane, add_bias=False)


def wrapNN(f,*args,**kwargs):
    def layer(x, scope='', is_training=True, reuse=None):
        return f(x,*args,**kwargs)
    return layer

def Dropout(p, name='Dropout'):
    def dropout_layer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
            # def drop(): return tf.nn.dropout(x,p)
            # def no_drop(): return x
            # return tf.cond(is_training, drop, no_drop)
            if is_training:
                return tf.nn.dropout(x,p)
            else:
                return x
    return dropout_layer

def ReLU(name='ReLU'):
    def layer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
            return tf.nn.relu(x)
    return layer

def HardTanh(name='HardTanh'):
    def layer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
            return tf.clip_by_value(x,-1,1)
    return layer


def View(shape, name='View', reuse=None):
    with tf.variable_scope(name, values=[x], reuse=reuse):
        return wrapNN(tf.reshape,shape=shape)

def SpatialMaxPooling(kW, kH=None, dW=None, dH=None, padding='VALID',
            name='SpatialMaxPooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def max_pool(x,is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
              return tf.nn.max_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return max_pool

def SpatialAveragePooling(kW, kH=None, dW=None, dH=None, padding='VALID',
        name='SpatialAveragePooling'):
    kH = kH or kW
    dW = dW or kW
    dH = dH or kH
    def avg_pool(x,is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
              return tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding=padding)
    return avg_pool
    
def BatchNormalization(name='BatchNormalization'):
    def batch_norm(x, is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
            return tf.layers.batch_normalization(x, training=is_training, reuse=reuse)
    return batch_norm

def LocalResposeNormalize(radius, alpha, beta, bias=1.0, name='LocalResposeNormalize'):
    def layer(x,is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
            return tf.nn.local_response_normalization(x, depth_radius = radius,
                                                      alpha = alpha, beta = beta,
                                                      bias = bias, name = name)
    return layer

def Sequential(moduleList):
    global regularizer
    if regularizer == None:
        if FLAGS.regularizer == 'L1':
            regularizer = tf.contrib.layers.l1_regularizer(FLAGS.regularizer_norm)
        elif FLAGS.regularizer == 'L2':
            regularizer = tf.contrib.layers.l2_regularizer(FLAGS.regularizer_norm)
    def model(x, is_training=True, reuse=None):
    # Create model
        output = x
        #with tf.variable_op_scope([x], None, name):
        for i,m in enumerate(moduleList):
            output = m(output, is_training=is_training, reuse=reuse)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, output)
        return output
    return model

def Concat(moduleList, dim=3):
    def model(x, is_training=True, reuse=None):
    # Create model
        outputs = []
        for i,m in enumerate(moduleList):
            name = 'layer_'+str(i)
            with tf.variable_scope(name, 'Layer', values=[x],  reuse=reuse):
                outputs[i] = m(x, is_training=is_training)
            output = tf.concat(dim, outputs)
        return output
    return model

def Residual(moduleList, connect=True, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True, reuse=None):
    # Create model
        with tf.variable_scope(name, values=[x]):
            if connect:
                output = tf.add(m(x, is_training=is_training, reuse=reuse), x)
            else:
                output = m(x, is_training=is_training, reuse=reuse)
            return output
    return model