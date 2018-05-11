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
tf.app.flags.DEFINE_integer('decay_epochs', -1,
                            """Iterations after which learning rate decays.""")
tf.app.flags.DEFINE_integer('test_epochs', 1,
                            """Iterations after which test.""")
tf.app.flags.DEFINE_integer('decay_plan', 0,
                            """the plan to decay laearning rate.""")
tf.app.flags.DEFINE_float('grad_clip_norm', 1e20,
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
tf.app.flags.DEFINE_string('dataset', 'None',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_boolean('summary', False,
                           """Record summary.""")
tf.app.flags.DEFINE_string('log_file', timestr+'.log',
                           'The log file name')
tf.app.flags.DEFINE_string('optimizer', 'NONE',
                           """optimizer for algorithms:MomentumOptimizer(MOM),
                           GradientDescentOptimizer(SGD), AdamOptimizer(ADA)""")
tf.app.flags.DEFINE_integer('num_gpu', 1,
                               """number of gpus to use.If 0 then cpu""")
tf.app.flags.DEFINE_integer('shift_gpu', 0,
                               """number of gpus to use.If 0 then cpu""")
tf.app.flags.DEFINE_boolean('debug', False,
                           """if debug.""")
tf.app.flags.DEFINE_string('weights_initial_file', '',
                           """if weights_initial.""")

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
            logger.info('Activaetion Name: ' + activation.name +
                ', Shape:' + str(activation.get_shape().as_list()))
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

def __init_with_file(weights_initial_file):
    assign_list = []
    trainable_variables = tf.trainable_variables()
    weights = np.load(weights_initial_file)['arr_0']
    flag=True
    for i in range(len(trainable_variables)):
        if trainable_variables[i].name.find('batch') != -1:
            if flag:
                weight = weights[i+1]
            else:
                 weight = weights[i-1]
            flag = not flag
        elif trainable_variables[i].name.find('conv') != -1:
            weight = np.transpose(weights[i], (2, 3, 1, 0))
        elif trainable_variables[i].name.find('fc') != -1:
            weight = np.transpose(weights[i], (1, 0))
        assign_list.append(tf.assign(trainable_variables[i], weight))


def __eval(model, logger):
    test_acc1, test_acc5, test_loss = evaluate(
        model, FLAGS.dataset, batch_size=FLAGS.batch_size//2, if_debug=FLAGS.debug,
        checkpoint_dir=FLAGS.checkpoint_dir, num_gpu=FLAGS.num_gpu)
    logger.info('Test Accuracy Top-1: %.3f' % test_acc1)
    logger.info('Test Accuracy Top-5: %.3f' % test_acc5)
    logger.info('Test Loss: %.3f' % test_loss)

def __close_sess(sess, coord, threads):
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    sess.close()

def __open_sess():
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    return sess, coord, threads

def __train(model, logger):

    # tf Graph input
    data = get_data_provider(FLAGS.dataset, training=True)

    num_image_per_epoch = data.size[0]
    if FLAGS.decay_epochs > 0:
        decay_step = tf.cast((num_image_per_epoch // FLAGS.batch_size) * FLAGS.decay_epochs, tf.int64)
    else:
        decay_step = 1e10

    global_step =  tf.train.get_or_create_global_step()
    if FLAGS.decay_plan == 0:
        lr = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, decay_step,
            FLAGS.learning_rate_decay_factor, staircase=True)
    elif FLAGS.decay_plan == 1:
        lr = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, decay_step,
            FLAGS.learning_rate_decay_factor, staircase=False)
    elif FLAGS.decay_plan == 2:
        lr = tf.train.piecewise_constant(
            global_step,
            [decay_step, decay_step + decay_step//4, decay_step + decay_step//2],
            [FLAGS.initial_learning_rate, FLAGS.initial_learning_rate/10, 
            FLAGS.initial_learning_rate/100, FLAGS.initial_learning_rate/1000])
    elif FLAGS.decay_plan == 3:
        lr = tf.train.piecewise_constant(
            global_step,
            [decay_step, decay_step + decay_step//2],
            [FLAGS.initial_learning_rate, FLAGS.initial_learning_rate/10,
            FLAGS.initial_learning_rate/100])
    else:
        assert 0, 'unrecognized decay_plan: '+str(FLAGS.decay_plan)

    if FLAGS.optimizer == 'SGD':
        opt = tf.train.GradientDescentOptimizer(lr)
    elif FLAGS.optimizer == 'MOM':
        opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
    elif FLAGS.optimizer == 'ADA':
        opt = tf.train.AdamOptimizer(lr)
    else:
        assert 0, 'unrecognized optimizer: '+FLAGS.optimizer
        
    assert FLAGS.num_gpu > 0, 'must use gpu'
    assert FLAGS.batch_size % FLAGS.num_gpu == 0, 'Batch size must be divisible by number of GPUs'
    with tf.name_scope('data'):
        x, yt = data.generate_batches(FLAGS.batch_size)

    x_splits = tf.split(
        axis=0, num_or_size_splits=FLAGS.num_gpu, value=x)
    yt_splits = tf.split(
        axis=0, num_or_size_splits=FLAGS.num_gpu, value=yt)
    tower_grads = []
    reuse = False
    for i in range(FLAGS.num_gpu):
        if FLAGS.debug:
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
            tf.cast(tf.nn.in_top_k(y, yt_splits[i], 1), tf.float32), name='accuracies')
        tf.add_to_collection('accuracies', accuracy)
    with tf.device('/cpu:0'):
        grads = __average_gradients(tower_grads)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    loss = tf.reduce_mean(tf.get_collection('total_losses'),name='total_loss')
    accuracy = tf.reduce_mean(tf.get_collection('accuracies'),name='accuracy')
        

    ema = tf.train.ExponentialMovingAverage(
        0.997, global_step, name='average')
    ema_op = ema.apply([loss, accuracy])
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

    loss_avg = ema.average(loss)
    accuracy_avg = ema.average(accuracy)
    
    updates_collection = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies([train_op]):
        train_op = tf.group(*updates_collection)

    if FLAGS.summary:
        tf.summary.scalar('loss/training', loss_avg)
        tf.summary.scalar('accuracy/training', accuracy_avg)
        __add_summaries(scalar_list=[accuracy, accuracy_avg, loss, loss_avg],
            activation_list=tf.get_collection(tf.GraphKeys.ACTIVATIONS),
            var_list=tf.trainable_variables(),
            grad_list=grads)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir)

    saver = tf.train.Saver(max_to_keep=5)
    # Configure options for session
    sess, coord, threads = __open_sess()
    epoch = FLAGS.ckpt_epoch

    if FLAGS.ckpt_file != '':
        ckpt_file_name = os.path.join(FLAGS.checkpoint_dir, FLAGS.ckpt_file)
        if epoch == 0:
            saver_init = tf.train.Saver(tf.trainable_variables())
            saver_init.restore(sess, ckpt_file_name)
        else:
            __close_sess(sess, coord, threads)
            __eval(model, logger)
            sess, coord, threads = __open_sess()
            saver.restore(sess, ckpt_file_name)
        sess.run(tf.assign(global_step, epoch * (num_image_per_epoch // FLAGS.batch_size)))
    elif FLAGS.weights_initial_file != '' and FLAGS.model == 'resnet18v3':
        assign_list = __init_with_file(FLAGS.weights_initial_file)
        sess.run(assign_list)
        ckpt_file_name = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        ckpt_file_name = saver.save(sess, save_path=ckpt_file_name, global_step=global_step)
        __close_sess(sess, coord, threads)
        __eval(model, logger)
        sess, coord, threads = __open_sess()
        saver.restore(sess, ckpt_file_name)

    if FLAGS.debug:
        __count_params(tf.trainable_variables(), tf.get_collection(tf.GraphKeys.ACTIVATIONS))


    while epoch != FLAGS.num_epochs:
        epoch += 1
        curr_count = 0

        logger.info('Started epoch %d' % epoch)
        try:
            while curr_count < data.size[0] and not coord.should_stop():
                curr_step, _, loss_val = sess.run([global_step, train_op, loss])
                curr_count += FLAGS.batch_size
                if curr_step % 100 == 0:
                    logger.info('Cunrrent step: %d, Loss: %.3f' % (curr_step, loss_val))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            break

        if FLAGS.summary:
            step, acc_value, loss_value, summary = sess.run(
                [global_step, accuracy_avg, loss_avg, summary_op])
        else:
            acc_value, loss_value = sess.run([accuracy_avg, loss_avg])
        ckpt_file_name = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
        ckpt_file_name = saver.save(sess, save_path=ckpt_file_name, global_step=global_step)
        logger.info('Finished epoch %d' % epoch)
        logger.info('Training Accuracy: %.3f' % acc_value)
        logger.info('Training Loss: %.3f' % loss_value)
        if FLAGS.summary:
            summary_writer.add_summary(summary, step)

        if epoch % FLAGS.test_epochs == 0:
            __close_sess(sess, coord, threads)
            __eval(model, logger)
            sess, coord, threads = __open_sess()
            saver.restore(sess, ckpt_file_name)
            

    # When done, ask the threads to stop.
    coord.request_stop()
    coord.join(threads)
    coord.clear_stop()
    sess.close()
    if FLAGS.summary:
        summary_writer.close()


def main(argv=None):  # pylint: disable=unused-argument
    FLAGS.checkpoint_dir = os.path.join(os.getcwd(), 'results', FLAGS.model + '_' + FLAGS.save)
    FLAGS.log_dir = os.path.join(FLAGS.checkpoint_dir , 'log')

    if not gfile.Exists(FLAGS.checkpoint_dir):
        gfile.MakeDirs(FLAGS.checkpoint_dir)
    model_file = os.path.join('./models', FLAGS.model + '.py')
    assert gfile.Exists(model_file), 'no model file named: ' + model_file
    gfile.Copy(model_file, FLAGS.checkpoint_dir + '/model.py', overwrite=True)
    if not gfile.Exists(FLAGS.log_dir):
        gfile.MakeDirs(FLAGS.log_dir)

    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('''%(asctime)s - %(levelname)s -: %(message)s''')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    handler = logging.FileHandler(os.path.join(FLAGS.log_dir,FLAGS.log_file), 'w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if FLAGS.debug or FLAGS.ckpt_epoch == 0:
        for key, value in sorted(FLAGS.__flags.items()):
            logger.info('%s: %s' % (key, value))
    m = importlib.import_module('models.' + FLAGS.model)
    __train(m.model, logger)


if __name__ == '__main__':
    tf.app.run()
