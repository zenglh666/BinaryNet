from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf
from data import get_data_provider

def evaluate(model, dataset,
        batch_size=128,
        num_gpu=1,
        if_debug=False,
        checkpoint_dir='./checkpoint'):
    with tf.Graph().as_default() as g:
        data = get_data_provider(dataset, training=False)
        with tf.device('/cpu:0'):
            x, yt = data.generate_batches(batch_size)

        # Build the Graph that computes the logits predictions
        if num_gpu <= 0:
            device_str='/cpu:0'
            with tf.device(device_str):
                y = model(x, is_training=True)
                accuracy = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(y, yt, 1), tf.float32))
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt, logits=y))
        else:
            assert batch_size % num_gpu == 0, ('Batch size must be divisible by number of GPUs')
            x_splits = tf.split(
                axis=0, num_or_size_splits=num_gpu, value=x)
            yt_splits = tf.split(
                axis=0, num_or_size_splits=num_gpu, value=yt)
            reuse = False

            for i in range(num_gpu):
                if if_debug:
                    device_str = '/gpu:0'
                else:
                    device_str = '/gpu:'+str(i)
                with tf.device(device_str):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        y = model(x_splits[i], is_training=True, reuse=reuse)
                        cross_entropy_loss = tf.reduce_mean(
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=yt_splits[i], logits=y),
                            name='cross_entropy_losses')
                        regularizer_loss = tf.reduce_sum(
                            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                            name='regularize_losses')
                        total_loss = tf.add(cross_entropy_loss, regularizer_loss,
                            name='total_losses')
                        tf.add_to_collection('total_losses', total_loss)
                accuracy_top1 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(y, yt_splits[i], 1), tf.float32),
                    name='accuracies_top1')
                tf.add_to_collection('accuracies_top1', accuracy_top1)
                accuracy_top5 = tf.reduce_mean(
                    tf.cast(tf.nn.in_top_k(y, yt_splits[i], 5), tf.float32),
                    name='accuracies_top5')
                tf.add_to_collection('accuracies_top5', accuracy_top5)

            loss = tf.reduce_mean(tf.get_collection('total_losses'),
                    name='total_loss')
            accuracy_top1 = tf.reduce_mean(tf.get_collection('accuracies_top1'),
                name='accuracy_top1')
            accuracy_top5 = tf.reduce_mean(tf.get_collection('accuracies_top5'),
                name='accuracy_top5')

        # Restore the moving average version of the learned variables for eval.
        #variable_averages = tf.train.ExponentialMovingAverage(
        #    MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver()#variables_to_restore)


        # Configure options for session
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(
                config=tf.ConfigProto(
                            log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options=gpu_options,
                            )
                        )
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir+'/')
        if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return

         # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                       start=True))

            num_batches = int(math.ceil(data.size[0] / batch_size))
            total_acc1 = 0  # Counts the number of correct predictions per batch.
            total_acc5 = 0  # Counts the number of correct predictions per batch.
            total_loss = 0 # Sum the loss of predictions per batch.
            step = 0
            while step < num_batches and not coord.should_stop():
                acc_val1, acc_val5, loss_val = sess.run([accuracy_top1, accuracy_top5, loss])
                total_acc1 += acc_val1
                total_acc5 += acc_val5
                total_loss += loss_val
                step += 1

            # Compute precision and loss
            total_acc1 /= num_batches
            total_acc5 /= num_batches
            total_loss /= num_batches

        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads)
        return total_acc1, total_acc5, total_loss

def main(argv=None):  # pylint: disable=unused-argument
  evaluate()


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('checkpoint_dir', './results/model',
                             """Directory where to read model checkpoints.""")
  tf.app.flags.DEFINE_string('dataset', 'cifar10',
                             """Name of dataset used.""")
  tf.app.flags.DEFINE_string('model_name', 'model',
                             """Name of loaded model.""")
  tf.app.flags.DEFINE_string('num_gpu', 0,
                               """number of gpus to use.If 0 then cpu""")

  FLAGS.log_dir = FLAGS.checkpoint_dir+'/log/'
      # Build the summary operation based on the TF collection of Summaries.
      # summary_op = tf.merge_all_summaries()

      # summary_writer = tf.train.SummaryWriter(log_dir)
          # summary = tf.Summary()
          # summary.ParseFromString(sess.run(summary_op))
          # summary.value.add(tag='accuracy/test', simple_value=precision)
          # summary_writer.add_summary(summary, global_step)

  tf.app.run()
