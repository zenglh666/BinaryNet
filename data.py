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
tf.app.flags.DEFINE_string('cifar_data_dir', 'F:\\data\\Cifar',
                           """Path to the cifar data directory.""")
tf.app.flags.DEFINE_integer('resize_size', 0,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('crop_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_boolean('style_th',False,
                            '''If we distort color''')
tf.app.flags.DEFINE_boolean('distort_color',False,
                            '''If we distort color''')
tf.app.flags.DEFINE_boolean('multiple_scale',False,
                            '''If we distort color''')
tf.app.flags.DEFINE_boolean('random_scale',False,
                            '''If we distort color''')
tf.app.flags.DEFINE_boolean('use_pickle',False,
                            '''If we distort color''')

FLAGS = tf.app.flags.FLAGS
train_data = None
train_labels = None
validation_data = None
validation_labels = None

def __read_cifar(filenames, train=False):
    global size_list, default_resize_size
    size_list = [28, 32, 36, 40, 44]
    if FLAGS.resize_size > 0:
        default_resize_size = FLAGS.resize_size
    else:
        default_resize_size = 32

    for f in filenames:
        assert tf.gfile.Exists(f), 'Failed to find file: ' + f
    
    data = []
    labels = []
    for file in filenames:
        fo = open(file, 'rb')
        entry = pickle.load(fo, encoding='bytes')
        data.append(entry[b'data'])
        if b'labels' in entry:
            labels += entry[b'labels']
        else:
            labels += entry[b'fine_labels']
        fo.close()
    if train:
        data = np.concatenate(data)
        data = data.reshape((50000, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        labels = np.array(labels).reshape((50000, 1))
    else:
        data = data[0]
        data = data.reshape((10000, 3, 32, 32))
        data = data.transpose((0, 2, 3, 1))  # convert to HWC
        labels = np.array(labels).reshape((10000, 1))

    data_tensor = tf.constant(data, dtype=tf.float32)
    labels_tensor = tf.constant(labels, dtype=tf.int32)

    image, label = tf.train.slice_input_producer([data_tensor, labels_tensor])
    return tf.image.convert_image_dtype(image, dtype=tf.float32), label

def __parse_example_proto(example_serialized):
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

def __create_pickle_data(filenames, train):
    global train_data, train_labels, validation_data, validation_labels
    if train:
        if train_data and train_labels:
            return
    else:
        if validation_data and validation_labels:
            return
    data = []
    labels = []
    for file in filenames:
        fo = open(file, 'rb')
        entry = pickle.load(fo, encoding='bytes')
        data.extend(entry['images'])
        labels.extend(entry['labels'])
        fo.close()
    assert len(data) == len(labels), 'data files are damaged'
    if train:
        train_data = data
        train_labels = labels
    else:
        assert len(data) == 50000
        validation_data = data
        validation_labels = labels
    return

def __get_train_data_label(index):
    return train_data[index], train_labels[index]

def __get_validation_data_label(index):
    return validation_data[index], validation_labels[index]

def __read_imagenet(data_files, name, train=True, num_readers=4):
    global size_list, mean, std, default_resize_size
    size_list = [224, 256, 288, 320, 352]
    mean = [ 0.485, 0.456, 0.406 ]
    std = [ 0.229, 0.224, 0.225 ]
    if FLAGS.resize_size > 0:
        default_resize_size = FLAGS.resize_size
    else:
        default_resize_size = 256

    if FLAGS.use_pickle:
        __create_pickle_data(data_files, train)
        if train:
            index_queue = tf.train.range_input_producer(limit=1281167, capacity=1024, shuffle=True)
            index = index_queue.dequeue()
            image_buffer, label_index = tf.py_func(__get_train_data_label, [index], [tf.string, tf.int32])
        else:
            index_queue = tf.train.range_input_producer(limit=50000, capacity=1024, shuffle=False)
            index = index_queue.dequeue()
            image_buffer, label_index = tf.py_func(__get_validation_data_label, [index], [tf.string, tf.int32])
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, tf.reshape(label_index, [1])

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

    if FLAGS.style_th:
        image_buffer, label_index = __parse_scale_example_proto(example_serialized)
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    else:
        image_buffer, label_index, bbox, class_text = __parse_example_proto(example_serialized)
        image = tf.image.decode_jpeg(image_buffer, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        if train:
            sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=bbox,
                min_object_covered=0.1,
                aspect_ratio_range=[0.75, 1.33],
                area_range=[0.75, 1.0],
                max_attempts=100,
                use_image_if_no_bounding_boxes=True)
            bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box
            image = tf.slice(image, bbox_begin, bbox_size)
        label_index = tf.subtract(label_index, 1)

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
        return images, tf.reshape(label_batch, [batch_size])



    def __preprocess_evaluation(self, img, height, width, normalize=None):
        img_size = tf.shape(img)
        img_size_float = tf.cast(img_size, tf.float32)
        resize_size_float = float(default_resize_size)
        resize_size_int = int(default_resize_size)
        if FLAGS.style_th:
            size = tf.cond(img_size[0] > img_size[1], 
                lambda: [tf.cast(img_size_float[0] / img_size_float[1] * resize_size_float, tf.int32),
                         resize_size_int],
                lambda: [resize_size_int,
                         tf.cast(img_size_float[1] / img_size_float[0] * resize_size_float, tf.int32)])
        else:
            size = [resize_size_int, resize_size_int]
        img = tf.image.resize_images(img, size)

        preproc_image = tf.image.resize_image_with_crop_or_pad(img, height, width)
        preproc_image.set_shape([height, width, 3])
        if normalize:
            # Subtract off the mean and divide by the variance of the pixels.
            preproc_image = tf.image.per_image_standardization(preproc_image)

        if FLAGS.style_th:
            mean_tensor = tf.reshape(tf.constant(mean, tf.float32), [1, 1, 3])
            std_tensor = tf.reshape(tf.constant(std, tf.float32), [1, 1, 3])
            preproc_image = tf.divide(tf.subtract(preproc_image, mean_tensor), std_tensor)
        else:
            preproc_image = tf.subtract(preproc_image, 0.5)
            preproc_image = tf.multiply(preproc_image, 2.0)

        return preproc_image

    def __preprocess_training(self, img, height, width, normalize=None):
        if FLAGS.style_th:
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
            size = tf.cond(img_size[0] > img_size[1], 
                lambda: [tf.cast(img_size_float[0] / img_size_float[1] * resize_size_float, tf.int32), 
                         resize_size_int],
                lambda: [resize_size_int, 
                         tf.cast(img_size_float[1] / img_size_float[0] * resize_size_float, tf.int32)])
            img = tf.image.resize_images(img, size)
            distorted_image = tf.random_crop(img, [height, width, 3])
        else:
            size = [height, width]
            distorted_image = tf.image.resize_images(img, size)
    
        distorted_image.set_shape([height, width, 3])

        if FLAGS.distort_color:
            distorted_image = tf.image.random_brightness(distorted_image,max_delta=0.25)
            distorted_image = tf.image.random_contrast(distorted_image,lower=0.6, upper=1.4)
            distorted_image = tf.image.random_saturation(distorted_image,lower=0.6, upper=1.4)
            distorted_image = tf.image.random_hue(distorted_image, max_delta=0.2)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        if normalize:
            # Subtract off the mean and divide by the variance of the pixels.
            distorted_image = tf.image.per_image_standardization(distorted_image)

        distorted_image = tf.image.random_flip_left_right(distorted_image)
        if FLAGS.style_th:
            mean_tensor = tf.reshape(tf.constant(mean, tf.float32), [1, 1, 3])
            std_tensor = tf.reshape(tf.constant(std, tf.float32), [1, 1, 3])
            distorted_image = tf.divide(tf.subtract(distorted_image, mean_tensor), std_tensor)
        else:
            distorted_image = tf.subtract(distorted_image, 0.5)
            distorted_image = tf.multiply(distorted_image, 2.0)

        return distorted_image

def group_batch_images(x):
    sz = x.get_shape().as_list()
    num_cols = int(math.sqrt(sz[0]))
    img = tf.slice(x, [0,0,0,0],[num_cols ** 2, -1, -1, -1])
    img = tf.batch_to_space(img, [[0,0],[0,0]], num_cols)

    return img



def get_data_provider(name, training=True):
    if name == 'imagenet' or name == 'imagenet_scale' or name == 'imagenet_pickle':
        if name == 'imagenet_scale':
            FLAGS.style_th = True
        if name == 'imagenet_pickle':
            FLAGS.style_th = True
            FLAGS.use_pickle = True
        if training:
            if FLAGS.imagenet_train_data_dir == '':
                if FLAGS.use_pickle:
                    path = os.path.join(FLAGS.imagenet_data_dir,'train_pickle')
                else:
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
                if FLAGS.use_pickle:
                    path = os.path.join(FLAGS.imagenet_data_dir,'validation_pickle')
                else:
                    path = os.path.join(FLAGS.imagenet_data_dir,'validation_tfrecord')
            else:
                path = FLAGS.imagenet_valid_data_dir
            file_pattern = os.path.join(path, '%s-*' % 'validation')
            data_files = tf.gfile.Glob(file_pattern)
            assert data_files, 'No files found for dataset %s/%s at %s' %(
                               self.name, 'validation', path)
            return DataProvider(__read_imagenet(data_files, name, train=training),
                                [50000, FLAGS.crop_size, FLAGS.crop_size, 3], training=training)

    elif name == 'cifar10':
        path = os.path.join(FLAGS.cifar_data_dir,'cifar10-python')
        data_dir = os.path.join(path, 'cifar-10-batches-py/')
        if training:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'data_batch_%d' % i)
                                    for i in range(1, 6)], train=training),
                                [50000, 24,24,3], training=training)
        else:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'test_batch')], train=training),
                                [10000, 24,24, 3], training=training)
    elif name == 'cifar100':
        path = os.path.join(FLAGS.cifar_data_dir,'cifar100-python')
        data_dir = os.path.join(path, 'cifar-100-py/')
        if training:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'train')], train=training),
                                [50000, 24,24,3], training=training)
        else:
            return DataProvider(__read_cifar([os.path.join(data_dir, 'test')], train=training),
                                [10000, 24,24, 3], training=training)
