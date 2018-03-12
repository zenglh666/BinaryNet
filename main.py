import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
import sys
import time

timestr = '-'.join(str(x) for x in list(tuple(datetime.now().timetuple())[:6]))
MOVING_AVERAGE_DECAY = 0.997
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', -1,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_integer('initial_learning_rate', 0.01,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('decay_step', 100000,
                            """Iterations after which learning rate decays.""")
tf.app.flags.DEFINE_string('model', 'model',
                           """Name of loaded model.""")
tf.app.flags.DEFINE_string('save', timestr,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('load', None,
                           """Name of loaded dir.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('gpu', False,
                           """use gpu.""")
tf.app.flags.DEFINE_string('device', 0,
                           """which gpu to use.""")
tf.app.flags.DEFINE_string('summary', True,
                           """Record summary.""")
tf.app.flags.DEFINE_string('log', 'ERROR',
                           'The threshold for what messages will be logged '
                            """DEBUG, INFO, WARN, ERROR, or FATAL.""")
tf.app.flags.DEFINE_string('optimizer', 'SGD',
                           """optimizer for algorithms:MomentumOptimizer(MOM),
                           GradientDescentOptimizer(SGD), AdamOptimizer(ADA)""")
tf.app.flags.DEFINE_string('regularizer', 'None',
                           """regularizer for weights:L1(L1),L2(L2)""")

FLAGS.checkpoint_dir = './results/' + FLAGS.save
FLAGS.log_dir = FLAGS.checkpoint_dir + '/log/'
# tf.logging.set_verbosity(FLAGS.log)

def count_params(var_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
    return num


def add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[], images=None):

    for var in scalar_list:
        if var is not None:
            tf.summary.scalar(var.op.name, var)

    for grad, var in grad_list:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in var_list:
        if var is not None:
            tf.summary.histogram(var.op.name, var)
            sz = var.get_shape().as_list()
            if len(sz) == 4 and sz[2] == 3:
                kernels = tf.transpose(var, [3, 0, 1, 2])
                tf.summary.image(var.op.name + '/kernels',
                                 group_batch_images(kernels), max_outputs=1)
    for activation in activation_list:
        if activation is not None:
            tf.summary.histogram(activation.op.name +
                                 '/activations', activation)
            tf.summary.scalar(activation.op.name + '/sparsity', tf.nn.zero_fraction(activation))
    if images is not None:
        images = tf.multiply(images, 0.5)
        images = tf.subtract(images, -0.5)
        tf.summary.image('/images', images)

def train(model, data,
          batch_size=128,
          log_dir='./log',
          checkpoint_dir='./checkpoint',
          num_epochs=-1):

    # tf Graph input
    with tf.device('/cpu:0'):
        with tf.name_scope('data'):
            x, yt = data.generate_batches(batch_size)

        global_step =  tf.contrib.framework.get_or_create_global_step()
        lr = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, FLAGS.decay_step,
            FLAGS.learning_rate_decay_factor, staircase=True)

    if FLAGS.gpu:
        device_str='/gpu:' + str(FLAGS.device)
    else:
        device_str='/cpu:0'
    with tf.device(device_str):
        y = model(x, is_training=True)
        with tf.name_scope('objective'):
            cross_entropy_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y),
                name='cross_entropy_loss')
            regularizer_loss = tf.reduce_sum(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                name='regularize_loss')
            loss = tf.add(cross_entropy_loss, regularizer_loss,
                name='total_loss')
            accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
        if FLAGS.optimizer == 'SGD':
            opt = tf.train.GradientDescentOptimizer(lr)
        elif FLAGS.optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        elif FLAGS.optimizer == 'ADA':
            opt = tf.train.AdamOptimizer(lr)
        else:
            assert True, 'unrecognized optimizer'
        train_op = opt.minimize(
            loss=loss,
            global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    tf.summary.scalar('loss/training', loss_avg)
    accuracy_avg = ema.average(accuracy)
    tf.summary.scalar('accuracy/training', accuracy_avg)

    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([train_op]):
        train_op = tf.group(*updates_collection)

    if FLAGS.summary:
        add_summaries( scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=tf.trainable_variables())
            #images=x)
            #grad_list=grads)

    summary_op = tf.summary.merge_all()

    # Configure options for session
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=gpu_options,
        )
    )
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_batches = data.size[0] / batch_size
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    epoch = 0

    print('num of trainable paramaters: %d' %
          count_params(tf.trainable_variables()))
    while epoch != num_epochs:
        epoch += 1
        curr_count = 0

        print('Started epoch %d' % epoch)
        while curr_count < data.size[0]:
            curr_step, _, loss_val = sess.run([global_step, train_op, loss])
            curr_count += FLAGS.batch_size
            if curr_step % 100 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S:   ", time.localtime())+
                    'Cunrrent step: %d, Loss: %.3f' % (curr_step, loss_val))
                sys.stdout.flush()

        step, acc_value, loss_value, summary = sess.run(
            [global_step, accuracy_avg, loss_avg, summary_op])
        saver.save(sess, save_path=checkpoint_dir +
                   '/model.ckpt', global_step=global_step)
        print(time.strftime("%Y-%m-%d %H:%M:%S:   ", time.localtime())+
            'Finished epoch %d' % epoch)
        print(time.strftime("%Y-%m-%d %H:%M:%S:   ", time.localtime())+
            'Training Accuracy: %.3f' % acc_value)
        print(time.strftime("%Y-%m-%d %H:%M:%S:   ", time.localtime())+
            'Training Loss: %.3f' % loss_value)

        test_acc, test_loss = evaluate(model, FLAGS.dataset,
                                       batch_size=batch_size,
                                       checkpoint_dir=checkpoint_dir)  # ,
        # log_dir=log_dir)
        print(time.strftime("%Y-%m-%d %H:%M:%S:   ", time.localtime())+
            'Test Accuracy: %.3f' % test_acc)
        print(time.strftime("%Y-%m-%d %H:%M:%S:   ", time.localtime())+
            'Test Loss: %.3f' % test_loss)
        sys.stdout.flush()

        summary_writer.add_summary(summary, step)

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
    if not gfile.Exists(FLAGS.checkpoint_dir):
        # gfile.DeleteRecursively(FLAGS.checkpoint_dir)
        gfile.MakeDirs(FLAGS.checkpoint_dir)
        model_file = os.path.join('models', FLAGS.model + '.py')
        assert gfile.Exists(model_file), 'no model file named: ' + model_file
        gfile.Copy(model_file, FLAGS.checkpoint_dir + '/model.py')
    m = importlib.import_module('results.' + FLAGS.save + '.model')
    data = get_data_provider(FLAGS.dataset, training=True)

    train(m.model, data,
          batch_size=FLAGS.batch_size,
          checkpoint_dir=FLAGS.checkpoint_dir,
          log_dir=FLAGS.log_dir,
          num_epochs=FLAGS.num_epochs)


if __name__ == '__main__':
    tf.app.run()
