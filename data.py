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

tf.app.flags.DEFINE_string('imagenet_train_data_dir', 'E:/data2/imagenet_train_tfrecord',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_string('imagenet_valid_data_dir', 'E:/data2/imagenet_valid_tfrecord',
                           """Path to the imagenet data directory.""")
tf.app.flags.DEFINE_string('cifar_data_dir', 'E:/data2/Cifar/',
                           """Path to the cifar data directory.""")
tf.app.flags.DEFINE_integer('resize_size', 256,
                            """Provide square images of this size.""")
FLAGS = tf.app.flags.FLAGS

def __read_cifar(filenames, cifar100=False, shuffle=True):
  """Reads and parses examples from CIFAR data files.
  """
  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)
  filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle)

  label_bytes = 1
  label_offset = 0
  if cifar100:
      label_offset = 1
  height = 32
  width = 32
  depth = 3
  image_bytes = height * width * depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + label_offset + image_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_bytes = tf.decode_raw(value, tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  label = tf.cast(
      tf.slice(record_bytes, [label_offset], [label_bytes]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record_bytes, [label_offset+label_bytes], [image_bytes]),
                           [depth, height, width])
  # Convert from [depth, height, width] to [height, width, depth].
  image = tf.transpose(depth_major, [1, 2, 0])

  return tf.image.convert_image_dtype(image, dtype=tf.float32), label

def __parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields:

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat(axis=0, values=[ymin, xmin, ymax, xmax])

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], label, bbox, features['image/class/text']


def __read_imagenet(data_files, train=True):
    if train:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)
    else:
      filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    reader = tf.TFRecordReader()
    key, value = reader.read(filename_queue)
    image_buffer, label_index, bbox, class_text = __parse_example_proto(value)
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    resize_image = tf.image.resize_images(image, [FLAGS.resize_size, FLAGS.resize_size])
    return resize_image, label_index - 1

class DataProvider:
    def __init__(self, data, size=None, training=True):
        self.size = size or [None]*4
        self.data = data
        self.training = training

    def generate_batches(self, batch_size, min_queue_examples=1000, num_threads=8):
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
            image_processed = preprocess_training(image, height=self.size[1], width=self.size[2])
            image_and_label = [image_processed, label]
            images, label_batch = tf.train.shuffle_batch(
            image_and_label,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
        else:
            image_processed = preprocess_evaluation(image, height=self.size[1], width=self.size[2])
            image_and_label = [image_processed, label]
            images, label_batch = tf.train.batch(
            image_and_label,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size)

        return images, tf.reshape(label_batch, [batch_size])



def preprocess_evaluation(img, height=None, width=None, normalize=None):
    img_size = img.get_shape().as_list()
    height = height or img_size[0]
    width = width or img_size[1]
    preproc_image = tf.image.resize_image_with_crop_or_pad(img, height, width)
    preproc_image = tf.subtract(preproc_image, 0.5)
    preproc_image = tf.multiply(preproc_image, 2.0)
    if normalize:
         # Subtract off the mean and divide by the variance of the pixels.
        preproc_image = tf.image.per_image_standardization(preproc_image)
    return preproc_image

def preprocess_training(img, height=None, width=None, normalize=None):
    img_size = img.get_shape().as_list()
    height = height or img_size[0]
    width = width or img_size[1]

    # Image processing for training the network. Note the many random
    # distortions applied to the image.

    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(img, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=32. / 255.)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.5, upper=1.5)

    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    if normalize:
        # Subtract off the mean and divide by the variance of the pixels.
        distorted_image = tf.image.per_image_standardization(distorted_image)
    return distorted_image

def group_batch_images(x):
    sz = x.get_shape().as_list()
    num_cols = int(math.sqrt(sz[0]))
    img = tf.slice(x, [0,0,0,0],[num_cols ** 2, -1, -1, -1])
    img = tf.batch_to_space(img, [[0,0],[0,0]], num_cols)

    return img



def get_data_provider(name, training=True):
    if name == 'imagenet':
        if training:
            tf_record_pattern = os.path.join(FLAGS.imagenet_train_data_dir, '%s-*' % 'train')
            data_files = tf.gfile.Glob(tf_record_pattern)
            assert data_files, 'No files found for dataset %s/%s at %s'%(
                               self.name, 'train', FLAGS.imagenet_train_data_dir)
            return DataProvider(__read_imagenet(data_files), [1281167, 227, 227, 3], True)
        else:
            tf_record_pattern = os.path.join(FLAGS.imagenet_valid_data_dir, '%s-*' % 'validation')
            data_files = tf.gfile.Glob(tf_record_pattern)
            assert data_files, 'No files found for dataset %s/%s at %s' %(
                               self.name, 'validation', FLAGS.imagenet_valid_data_dir)
            return DataProvider(__read_imagenet(data_files), [50000, 227, 227, 3], True)

    elif name == 'cifar10':
        path = os.path.join(FLAGS.cifar_data_dir,'cifar10')
        data_dir = os.path.join(path, 'cifar-10-batches-bin/')
        if training:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'data_batch_%d.bin' % i)
                                    for i in range(1, 6)], False, True),
                                [50000, 32,32,3], True)
        else:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'test_batch.bin')], False, False),
                                [10000, 32,32, 3], False)
    elif name == 'cifar100':
        path = os.path.join(FLAGS.cifar_data_dir,'cifar100')
        data_dir = os.path.join(path, 'cifar-100-binary/')
        if training:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'train.bin')], True, True),
                                [50000, 32,32,3], True)
        else:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'test.bin')], True, False),
                                [10000, 32,32, 3], False)
