from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import tensorflow as tf
import math
from six.moves import urllib
import csv
import glob
import re
import numpy as np
import pickle

tf.app.flags.DEFINE_string('imagenet_data_dir', 'F:\\data\\imagenet_scale256',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_string('imagenet_train_data_dir', '',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_string('imagenet_valid_data_dir', '',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_integer('resize_size', 0,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('crop_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_boolean('multiple_scale',False,
                            '''If we distort color''')
tf.app.flags.DEFINE_boolean('random_scale',False,
                            '''If we distort color''')

FLAGS = tf.app.flags.FLAGS

def __parse_scale_example_proto(example_serialized):
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
        'image/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
    }

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/label'], dtype=tf.int32)

    return features['image/encoded'], label

def __read_imagenet(data_files, name, train=True, num_readers=2):
    filename_queue = tf.train.string_input_producer(data_files, shuffle=False, capacity=32)
    # Approximate number of examples per shard.
    examples_per_shard = 1024
    # Size the random shuffle queue to balance between good global
    # mixing (more examples) and memory use (fewer examples).
    # 1 image uses 299*299*3*4 bytes = 1MB
    # The default input_queue_memory_factor is 16 implying a shuffling queue
    # size: examples_per_shard * 16 * 1MB = 17.6GB
    if train:
        examples_queue = tf.RandomShuffleQueue(
            capacity=examples_per_shard * 4,
            min_after_dequeue=examples_per_shard,
            dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
            capacity=examples_per_shard,
            dtypes=[tf.string])

    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = tf.TFRecordReader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

        tf.train.queue_runner.add_queue_runner(
            tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
        example_serialized = examples_queue.dequeue()
    else:
        reader = tf.TFRecordReader()
        _, example_serialized = reader.read(filename_queue)

    image_buffer, label_index = __parse_scale_example_proto(example_serialized)
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    return image, label_index

class DataProvider:
    def __init__(self, data, size=None, training=True):
        self.size = size or [None]*4
        self.data = data
        self.training = training

    def generate_batches(self, batch_size, min_queue_examples=1024, num_threads=4):
        """Construct a queued batch of images and labels.

        Args:
        image: 3-D Tensor of [height, width, 3] of type.float32.
        label: 1-D Tensor of type.int32
        min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
        batch_size: Number of images per batch.

        Returns:
        images: Images. 4D tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
        """
        # Create a queue that shuffles the examples, and then
        # read 'batch_size' images + labels from the example queue.

        image, label = self.data
        if self.training:
            image_processed = self.__preprocess_training(image, height=self.size[1], width=self.size[2])
            images, label_batch = tf.train.shuffle_batch(
                [image_processed, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=min_queue_examples * 4,
                min_after_dequeue=min_queue_examples)
        else:
            image_processed = self.__preprocess_evaluation(image, height=self.size[1], width=self.size[2])
            images, label_batch = tf.train.batch(
                [image_processed, label],
                batch_size=batch_size,
                num_threads=num_threads,
                capacity=min_queue_examples)

        with tf.device('/gpu:0'):
            mean_tensor = tf.reshape(tf.constant(mean, tf.float32), [1, 1, 1, 3])
            std_tensor = tf.reshape(tf.constant(std, tf.float32), [1, 1, 1, 3])
            images = tf.divide(tf.subtract(images, mean_tensor), std_tensor)
        return images, tf.reshape(label_batch, [batch_size])



    def __preprocess_evaluation(self, img, height, width):
        img_size = tf.shape(img)
        img_size_float = tf.cast(img_size, tf.float32)
        resize_size_float = float(default_resize_size)
        resize_size_int = int(default_resize_size)

        size = [resize_size_int, resize_size_int]
        img = tf.image.resize_images(img, size)

        preproc_image = tf.image.resize_image_with_crop_or_pad(img, height, width)
        preproc_image.set_shape([height, width, 3])

        return preproc_image

    def __preprocess_training(self, img, height, width, normalize=None):
        if FLAGS.multiple_scale:
            if FLAGS.random_scale:
                resize_size_int = tf.random_uniform([1], 
                    minval=size_list[0], maxval=size_list[-1], dtype=tf.int32)[0]
                resize_size_float = tf.cast(resize_size_int, tf.float32)
            else:
                size_list_float_tensor = tf.constant(size_list, tf.float32)
                size_list_int_tensor = tf.constant(size_list, tf.int32)
                size_index = tf.random_uniform([1], maxval=len(size_list), dtype=tf.int32)[0]
                resize_size_float = size_list_float_tensor[size_index]
                resize_size_int = size_list_int_tensor[size_index]
        else:
            resize_size_float = float(default_resize_size)
            resize_size_int = int(default_resize_size)
        img_size = tf.shape(img)
        img_size_float = tf.cast(img_size, tf.float32)
        size = [resize_size_int, resize_size_int]
        img = tf.image.resize_images(img, size)
        distorted_image = tf.random_crop(img, [height, width, 3])
        distorted_image.set_shape([height, width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        return distorted_image

def group_batch_images(x):
    sz = x.get_shape().as_list()
    num_cols = int(math.sqrt(sz[0]))
    img = tf.slice(x, [0,0,0,0],[num_cols ** 2, -1, -1, -1])
    img = tf.batch_to_space(img, [[0,0],[0,0]], num_cols)

    return img

def get_data_provider(name, training=True):
    if name == 'imagenet' or name == 'imagenet_scale':
        global size_list, mean, std, default_resize_size
        size_list = [224, 256, 288, 320, 352]
        mean = [ 0.485, 0.456, 0.406 ]
        std = [ 0.229, 0.224, 0.225 ]
        if FLAGS.resize_size > 0:
            default_resize_size = FLAGS.resize_size
        else:
            default_resize_size = 256
        if training:
            if FLAGS.imagenet_train_data_dir == '':
                path = os.path.join(FLAGS.imagenet_data_dir,'train_tfrecord')
            else:
                path = FLAGS.imagenet_train_data_dir
            file_pattern = os.path.join(path, '%s-*' % 'train')
            data_files = tf.gfile.Glob(file_pattern)
            assert data_files, 'No files found for dataset %s/%s at %s'%(
                               self.name, 'train', path)
            return DataProvider(__read_imagenet(data_files, name, train=training),
                                [1281167, FLAGS.crop_size, FLAGS.crop_size, 3], training=training)
        else:
            if FLAGS.imagenet_valid_data_dir == '':
                path = os.path.join(FLAGS.imagenet_data_dir,'validation_tfrecord')
            else:
                path = FLAGS.imagenet_valid_data_dir
            file_pattern = os.path.join(path, '%s-*' % 'validation')
            data_files = tf.gfile.Glob(file_pattern)
            assert data_files, 'No files found for dataset %s/%s at %s' %(
                               self.name, 'validation', path)
            return DataProvider(__read_imagenet(data_files, name, train=training),
                                [50000, FLAGS.crop_size, FLAGS.crop_size, 3], training=training)
