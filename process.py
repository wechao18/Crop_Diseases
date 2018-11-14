import glob
import json
import os
import random

import sys
import tensorflow as tf
import math

import tensorflow as tf
#from inception_resnet import AgriculturalDisease

slim = tf.contrib.slim


_RANDOM_SEED = 0
_NUM_SHARDS = 5

class ImageReader(object):
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def get_filenames_and_classes(dataset_dir):
    """Returns a list of filenames and inferred class names.
    Args:
      dataset_dir: A directory containing a set of subdirectories representing
        class names. Each subdirectory should contain PNG or JPG encoded images.
    Returns:
      A list of image file paths, relative to `dataset_dir` and the list of
      subdirectories, representing class names.
    """
    #AgriculturalDisease = os.path.join(dataset_dir, 'AgriculturalDisease_validationset')
    AgriculturalDisease = os.path.join(dataset_dir, 'AgriculturalDisease_trainingset')

    directories = []
    class_names = []

    for filename in os.listdir(AgriculturalDisease):
        path = os.path.join(AgriculturalDisease, filename)

        for filename1 in os.listdir(path):
            path1 = os.path.join(path, filename1)
            if os.path.isdir(path1):
                directories.append(path1)
                class_names.append((filename + filename1).replace(' ',''))
            else:
                directories.append(path)
                class_names.append(filename.replace(' ',''))
                break

    photo_filenames = []
    for directory in directories:
        file_glob = os.path.join(directory, '*.jpg')
        filelist = glob.glob(file_glob)

        for file in filelist:
            # path = os.path.join(directory, filename)
            photo_filenames.append(file)

    return photo_filenames, sorted(class_names)

def int64_feature(values):
  """Returns a TF-Feature of int64s.
  Args:
    values: A scalar or list of values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def float_feature(values):
  """Returns a TF-Feature of floats.
  Args:
    values: A scalar of list of values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'AgriculturalDisease_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def image_to_tfexample_test(image_data, image_format, image_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image_id': bytes_feature(image_id)
    }))


def convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        #image = tf.image.decode_jpeg(image)
                        one_dir = os.path.dirname(filenames[i])

                        one_class_name = os.path.basename(one_dir)
                        if one_class_name in ['一般', '严重']:
                            two_dir = os.path.dirname(one_dir)
                            two_class_name = os.path.basename(two_dir)
                            class_name = (two_class_name + one_class_name).replace(' ', '')
                        else:
                            class_name = one_class_name.replace(' ','')
                        print(class_name)
                        class_id = class_names_to_ids[class_name]


                        example = image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def write_label_file(labels_to_class_names, dataset_dir,
                     filename='labels.txt'):
  """Writes a file with the list of class names.
  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))

def json_convert_dataset(jsonpath, split_name, dataset_dir, image_path):
    assert split_name in ['train', 'validation']
    with open(jsonpath, encoding='utf-8') as f:
        json_data = json.load(f)
        random.seed(_RANDOM_SEED)
        random.shuffle(json_data)

        num_per_shard = int(math.ceil(len(json_data) / float(_NUM_SHARDS)))
        with tf.Graph().as_default():
            image_reader = ImageReader()
            with tf.Session('') as sess:

                for shard_id in range(_NUM_SHARDS):
                    output_filename = get_dataset_filename(
                        dataset_dir, split_name, shard_id)

                    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                        start_ndx = shard_id * num_per_shard
                        end_ndx = min((shard_id + 1) * num_per_shard, len(json_data))
                        for i in range(start_ndx, end_ndx):
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                                i + 1, len(json_data), shard_id))
                            sys.stdout.flush()

                            # Read the filename:
                            filename = os.path.join(image_path, json_data[i]['image_id'])
                            image_data = tf.gfile.FastGFile(filename, 'rb').read()
                            height, width = image_reader.read_image_dims(sess, image_data)
                            # image = tf.image.decode_jpeg(image)

                            class_id = json_data[i]['disease_class']

                            example = image_to_tfexample(
                                image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())

        sys.stdout.write('\n')
        sys.stdout.flush()
        #for data in json_data:
        #    image_id = data['image_id']
        #    class_id = data['disease_class']

def json_convert_dataset_test(split_name, dataset_dir, image_path):
    assert split_name in ['test']
    filenames = []
    extensions = ['jpg', 'JPG']
    for extension in extensions:
        file_glob = os.path.join(image_path, '*.' + extension)
        filenames.extend(glob.glob(file_glob))

    print(len(filenames))

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
    image_reader = ImageReader()
    for shard_id in range(_NUM_SHARDS):
        output_filename = get_dataset_filename(
            dataset_dir, split_name, shard_id)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()
                # Read the filename:
                image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                base_name = os.path.basename(filenames[i])
                image_id = os.path.splitext(base_name)[0]
                example = image_to_tfexample_test(
                    image_data, b'jpg', image_id.encode())
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


	
if __name__ == '__main__':
    #json_convert_dataset('./AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json', 'train', 'tf_data', './AgriculturalDisease_trainingset/images')

    #json_convert_dataset('./AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json', 'validation', 'tf_data', './AgriculturalDisease_validationset/images')
    json_convert_dataset_test('test', 'tf_data', './AgriculturalDisease_testA/images')

'''
    photo_filenames, class_names = get_filenames_and_classes('E:\datasets')
    #print(photo_filenames)
    print(class_names)
    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))
    print(len(class_names_to_ids))
    print(class_names_to_ids)
    convert_dataset('validation', photo_filenames, class_names_to_ids, './tf_data')
    # Finally, write the labels file:
    #labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    #write_label_file(labels_to_class_names, './tf_data')

    # Selects the 'validation' dataset.
    DATA_DIR = './tf_data'
    dataset = AgriculturalDisease.get_split('validation', DATA_DIR)

    # Creates a TF-Slim DataProvider which reads the dataset in the background
    # during both training and testing.
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset)
    [image, label] = provider.get(['image', 'label'])
    '''
