import tensorflow as tf
import importlib
import tensorflow.python.platform
import os
import numpy as np
from datetime import datetime
from tensorflow.python.platform import gfile
from data import *
from evaluate import evaluate
import nnUtils
import sys
import logging

timestr =  datetime.now().isoformat().replace(':','-').replace('.','MS')
MOVING_AVERAGE_DECAY = 0.997
FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 25,
                            """Number of epochs to train. -1 for unlimited""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('decay_epochs', 10,
                            """Iterations after which learning rate decays.""")
tf.app.flags.DEFINE_integer('test_epochs', 1,
                            """Iterations after which test.""")
tf.app.flags.DEFINE_float('grad_clip_norm', 1e1,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('momentum', 0.9,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_string('model', 'model',
                           """Name of loaded model.""")
tf.app.flags.DEFINE_string('save', timestr,
                           """Name of saved dir.""")
tf.app.flags.DEFINE_string('load', None,
                           """Name of loaded dir.""")
tf.app.flags.DEFINE_string('ckpt_file', '',
                           """Name of ckpt file.""")
tf.app.flags.DEFINE_integer('ckpt_epoch', 0,
                           """number of ckpt epoch.""")
tf.app.flags.DEFINE_string('dataset', 'cifar10',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_boolean('summary', False,
                           """Record summary.""")
tf.app.flags.DEFINE_string('log_file', timestr+'.log',
                           'The log file name')
tf.app.flags.DEFINE_string('optimizer', 'SGD',
                           """optimizer for algorithms:MomentumOptimizer(MOM),
                           GradientDescentOptimizer(SGD), AdamOptimizer(ADA)""")
tf.app.flags.DEFINE_integer('num_gpu', 1,
                               """number of gpus to use.If 0 then cpu""")
tf.app.flags.DEFINE_integer('shift_gpu', 0,
                               """number of gpus to use.If 0 then cpu""")
tf.app.flags.DEFINE_boolean('debug', False,
                           """if debug.""")

FLAGS.checkpoint_dir = './results/' + FLAGS.model + '_' + FLAGS.save
FLAGS.log_dir = FLAGS.checkpoint_dir + '/log/'
logger = logging.getLogger(__name__)
formatter = logging.Formatter('''%(asctime)s - %(name)s'''
    ''' - %(levelname)s -: %(message)s''')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(logging.DEBUG)

def __average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
      List of pairs of (gradient, variable) where the gradient has been averaged
      across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(axis=0, values=grads)
      grad = tf.reduce_mean(grad, 0)
      grad = tf.clip_by_norm(grad, FLAGS.grad_clip_norm)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

def __count_params(var_list, activation_list):
    num = 0
    for var in var_list:
        if var is not None:
            num += var.get_shape().num_elements()
            logger.info('Layer Name: ' + var.name + ', Shape:' + str(var.get_shape().as_list()))
    for activation in activation_list:
        if activation is not None:
            logger.info('Activaetion Name: ' + activation.name + ', Shape:' + str(activation.get_shape().as_list()))
    logger.info('num of trainable paramaters: %d' % num)


def __add_summaries(scalar_list=[], activation_list=[], var_list=[], grad_list=[], images=None):

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


def train(model, dataset, optimizer,
          batch_size=128,
          num_epochs=-1,
          initial_learning_rate=0.01,
          learning_rate_decay_factor=0.1,
          decay_epochs=-1,
          num_gpu=0,
          if_summary=True,
          if_debug=False,
          log_dir='./log',
          checkpoint_dir='./checkpoint'):

    # tf Graph input
    
    data = get_data_provider(dataset, training=True)

    num_image_per_epoch = data.size[0]
    if decay_epochs > 0:
        decay_step = (num_image_per_epoch // batch_size) * decay_epochs
    else:
        decay_step = 1e10
    with tf.device('/cpu:0'):
        with tf.name_scope('data'):
            x, yt = data.generate_batches(batch_size)

        global_step =  tf.train.get_or_create_global_step()

        if optimizer == 'SGD':
            lr = tf.train.exponential_decay(
                initial_learning_rate, global_step, decay_step,
                learning_rate_decay_factor, staircase=True)
            opt = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == 'MOM':
            lr = tf.train.exponential_decay(
                initial_learning_rate, global_step, decay_step,
                learning_rate_decay_factor, staircase=True)
            opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
        elif optimizer == 'ADA':
            lr = tf.train.exponential_decay(
                initial_learning_rate, global_step, decay_step,
                learning_rate_decay_factor, staircase=False)
            opt = tf.train.AdamOptimizer(lr)
        else:
            opt = None
        assert opt != None, 'unrecognized optimizer: '+optimizer

    if num_gpu <= 0:
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
            train_op = opt.minimize(
                loss=loss,
                global_step=global_step)
        grads = []
    else:
        assert batch_size % num_gpu == 0, ('Batch size must be divisible by number of GPUs')
        x_splits = tf.split(
            axis=0, num_or_size_splits=num_gpu, value=x)
        yt_splits = tf.split(
            axis=0, num_or_size_splits=num_gpu, value=yt)
        tower_grads = []
        reuse = False
        for i in range(num_gpu):
            if if_debug:
                device_str = '/gpu:0'
            else:
                device_str = '/gpu:'+str(i + FLAGS.shift_gpu)
            with tf.device(device_str):
                with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    y = model(x_splits[i], is_training=True, reuse=reuse)

                    with tf.name_scope('objective'):
                        cross_entropy_loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt_splits[i], logits=y),
                            name='cross_entropy_losses')
                        tf.add_to_collection('cross_entropy_losses', cross_entropy_loss)
                        regularizer_loss = tf.reduce_sum(
                            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                            name='regularize_losses')
                        total_loss = tf.add(cross_entropy_loss, regularizer_loss,
                            name='total_losses')
                        tf.add_to_collection('total_losses', total_loss)
                    reuse = True
                    grads = opt.compute_gradients(total_loss)
                    tower_grads.append(grads)
            accuracy = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(y, yt_splits[i], 1), tf.float32),
                name='accuracies')
            tf.add_to_collection('accuracies', accuracy)
        with tf.device('/cpu:0'):
          grads = __average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads, global_step=global_step)
        loss = tf.reduce_mean(tf.get_collection('total_losses'),
            name='total_loss')
        accuracy = tf.reduce_mean(tf.get_collection('accuracies'),
            name='accuracy')
        

    ema = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step, name='average')
    ema_op = ema.apply([loss, accuracy])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    accuracy_avg = ema.average(accuracy)
    
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([train_op]):
        train_op = tf.group(*updates_collection)

    if if_summary:
        tf.summary.scalar('loss/training', loss_avg)
        tf.summary.scalar('accuracy/training', accuracy_avg)
        __add_summaries( scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=tf.trainable_variables(),
            grad_list=grads)

    summary_op = tf.summary.merge_all()

    # Configure options for session
    sess = tf.Session(
        config=tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True),
        )
    )
    saver = tf.train.Saver(max_to_keep=5)

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    num_batches = data.size[0] / batch_size
    summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
    epoch = FLAGS.ckpt_epoch

    __count_params(tf.trainable_variables(), tf.get_collection(tf.GraphKeys.ACTIVATIONS))

    if FLAGS.ckpt_file != '':
        ckpt_file_name = os.path.join(checkpoint_dir, FLAGS.ckpt_file)
        saver.restore(sess, ckpt_file_name)
        if ckpt_file_name.find('..') != -1:
           sess.run(tf.assign(global_step,0))

    while epoch != num_epochs:
        epoch += 1
        curr_count = 0

        logger.info('Started epoch %d' % epoch)
        while curr_count < data.size[0]:
            curr_step, _, loss_val = sess.run([global_step, train_op, loss])
            curr_count += batch_size
            if curr_step % 100 == 0:
                logger.info('Cunrrent step: %d, Loss: %.3f' % (curr_step, loss_val))

        step, acc_value, loss_value, summary = sess.run(
            [global_step, accuracy_avg, loss_avg, summary_op])
        saver.save(sess, save_path=checkpoint_dir +
                   '/model.ckpt', global_step=global_step)
        logger.info('Finished epoch %d' % epoch)
        logger.info('Training Accuracy: %.3f' % acc_value)
        logger.info('Training Loss: %.3f' % loss_value)

        if epoch % FLAGS.test_epochs == 0:
          test_acc1, test_acc5, test_loss = evaluate(model, dataset,
                                                     batch_size=batch_size//2,
                                                     if_debug=if_debug,
                                                     checkpoint_dir=checkpoint_dir)  # ,
          # log_dir=log_dir)
          logger.info('Test Accuracy Top-1: %.3f' % test_acc1)
          logger.info('Test Accuracy Top-5: %.3f' % test_acc5)
          logger.info('Test Loss: %.3f' % test_loss)

          summary_writer.add_summary(summary, step)

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
    for key, value in FLAGS.__flags.items():
        logger.info('%s: %s' % (key, value))
    m = importlib.import_module('models.' + FLAGS.model)
    train(m.model, FLAGS.dataset, FLAGS.optimizer,
          batch_size=FLAGS.batch_size,
          num_epochs=FLAGS.num_epochs,
          initial_learning_rate=FLAGS.initial_learning_rate,
          learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
          decay_epochs=FLAGS.decay_epochs,
          num_gpu=FLAGS.num_gpu,
          if_summary=FLAGS.summary,
          if_debug=FLAGS.debug,
          log_dir=FLAGS.log_dir,
          checkpoint_dir=FLAGS.checkpoint_dir)


if __name__ == '__main__':
    
    if not gfile.Exists(FLAGS.checkpoint_dir):
        gfile.MakeDirs(FLAGS.checkpoint_dir)
        logger.warning('create direction: ' + FLAGS.checkpoint_dir)
    model_file = os.path.join('./models', FLAGS.model + '.py')
    assert gfile.Exists(model_file), 'no model file named: ' + model_file
    gfile.Copy(model_file, FLAGS.checkpoint_dir + '/model.py', overwrite=True)
    if not gfile.Exists(FLAGS.log_dir):
        gfile.MakeDirs(FLAGS.log_dir)
        logger.warning('create direction: ' + FLAGS.log_dir)
    
    handler = logging.FileHandler(os.path.join(FLAGS.log_dir,FLAGS.log_file), 'w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    tf.app.run()
