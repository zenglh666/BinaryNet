import tensorflow as tf
import math
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops

tf.app.flags.DEFINE_string('regularizer', 'None',
                           """regularizer for weights:L1(L1),L2(L2)""")
tf.app.flags.DEFINE_float('regularizer_norm', 0.000,
                          """regularizer norm.""")
tf.app.flags.DEFINE_integer('bit', 1,
                               """number of bit""")
tf.app.flags.DEFINE_float('zeta', 0.0,
                          """zeta.""")
tf.app.flags.DEFINE_boolean('weight_norm', False,
                           """weight norm.""")
FLAGS = tf.app.flags.FLAGS
regularizer = None
    
def binarize(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)

def BinarizedSpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=False, name='BinarizedSpatialConvolution'):
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
        padding='VALID', bias=False, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            bin_w = binarize(tf.clip_by_value(w,-1,1))
            out = tf.nn.conv2d(x, bin_w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def AccurateBinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=False, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            w = tf.clip_by_value(w,-1,1)
            w_res = tf.identity(w)
            w_apr = tf.zeros(w.get_shape())
            for i in range(FLAGS.bit):
                bin_w = binarize(w_res)
                alpha = tf.reduce_mean(tf.abs(w_res))
                w_mul = tf.multiply(bin_w, alpha)
                w_apr = tf.add(w_apr, w_mul)
                w_res = tf.subtract(w_res, w_mul)
            out = tf.nn.conv2d(x, w_apr, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def MoreAccurateBinarizedWeightOnlySpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=False, name='BinarizedWeightOnlySpatialConvolution'):
    '''
    This function is used only at the first layer of the model as we dont want to binarized the RGB images
    '''
    def bc_conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(mode='fan_avg'))
            if FLAGS.weight_norm:
                w = tf.layers.batch_normalization(w, axis=2, training=is_training, trainable=False, reuse=reuse, epsilon=1e-20)
            #else:
                #w = tf.clip_by_value(w,-1,1)

            w_pow = tf.pow(tf.abs(w),FLAGS.zeta)
            w_pow_sum = tf.reduce_sum(w_pow, axis=[0,1,2], keep_dims=True)
            w_res = tf.identity(w)
            w_apr = tf.zeros(w.get_shape())
            for i in range(FLAGS.bit):
                bin_w = binarize(w_res)
                if FLAGS.zeta < 0.1:
                    alpha = tf.reduce_mean(tf.abs(w_res), axis=[0,1,2], keep_dims=True)
                else:
                    alpha = tf.div(tf.reduce_mean(tf.multiply(w_pow, tf.abs(w_res)), axis=[0,1,2], keep_dims=True), w_pow_sum)
                w_mul = tf.multiply(bin_w, alpha)
                w_apr = tf.add(w_apr, w_mul)
                w_res = tf.subtract(w_res, w_mul)
            out = tf.nn.conv2d(x, w_apr, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return bc_conv2d

def SpatialConvolution(nOutputPlane, kW, kH, dW=1, dH=1,
        padding='VALID', bias=False, name='SpatialConvolution'):
    def conv2d(x, is_training=True, reuse=None):
        nInputPlane = x.get_shape().as_list()[3]
        with tf.variable_scope(name, values=[x], reuse=reuse):
            w = tf.get_variable('weight', [kH, kW, nInputPlane, nOutputPlane],
                            initializer=tf.variance_scaling_initializer(),
                            regularizer=regularizer)
            if FLAGS.weight_norm:
                w = tf.layers.batch_normalization(w, axis=2, training=is_training, trainable=False, reuse=reuse, epsilon=1e-20)
            out = tf.nn.conv2d(x, w, strides=[1, dH, dW, 1], padding=padding)
            if bias:
                b = tf.get_variable('bias', [nOutputPlane],initializer=tf.zeros_initializer)
                out = tf.nn.bias_add(out, b)
            return out
    return conv2d

def Affine(nOutputPlane, bias=False, name=None):
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

def BinarizedAffine(nOutputPlane, bias=False, name=None):
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

def BinarizedWeightOnlyAffine(nOutputPlane, bias=False, name=None):
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
            w = tf.clip_by_value(w,-1,1)
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

def HardTanhReLU(name='HardTanh'):
    def layer(x, is_training=True, reuse=None):
        with tf.variable_scope(name, values=[x], reuse=reuse):
            return tf.clip_by_value(x,0,1)
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
            return tf.layers.batch_normalization(x, training=is_training, reuse=reuse, momentum=0.997, epsilon=1e-5)
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

def ResidualV2(moduleList, connect=True, kW=2, kH=2, dW=2, dH=2, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True, reuse=None):
    # Create model
        with tf.variable_scope(name, values=[x]):
            if connect:
                output = tf.add(m(x, is_training=is_training, reuse=reuse), x)
            else:
                output = m(x, is_training=is_training, reuse=reuse)
                x = tf.nn.avg_pool(x, ksize=[1, kW, kH, 1], strides=[1, dW, dH, 1], padding='SAME')
                zero_shape = x.get_shape().as_list()
                zero_shape[-1] = output.get_shape().as_list()[-1] - zero_shape[-1]
                zero = tf.zeros(zero_shape)
                x = tf.concat([x, zero] , axis=-1)
                output = tf.add(output, x)
            return output
    return model

def ResidualV3(moduleList, connect=True, mapfunc=None, name='Residual'):
    m = Sequential(moduleList)
    def model(x, is_training=True, reuse=None):
    # Create model
        with tf.variable_scope(name, values=[x]):
            if connect:
                output = tf.add(m(x, is_training=is_training, reuse=reuse), x)
            elif mapfunc is not None:
                output = m(x, is_training=is_training, reuse=reuse)
                x_proj = mapfunc(x, is_training=is_training, reuse=reuse)
                batch_norm = BatchNormalization(name='rbatch_proj')
                x_proj = batch_norm(x_proj, is_training=is_training, reuse=reuse)
                output = tf.add(output, x_proj)
            else:
                output = m(x, is_training=is_training, reuse=reuse)
            return output

    return model