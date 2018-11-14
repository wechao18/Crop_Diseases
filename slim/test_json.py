# encoding:utf-8
import math

import tensorflow as tf

from matplotlib import pyplot as plt
from nets import inception
from preprocessing import inception_preprocessing
import numpy as np
import json
import os
from datasets import dataset_factory
from datasets import AgriculturalDisease
from preprocessing import preprocessing_factory
from nets import nets_factory


tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


slim = tf.contrib.slim


#batch_size = 1
#image_test = tf.placeholder(tf.float32, [batch_size, 299, 299, 3])
#label_test = tf.placeholder(tf.int32, [batch_size])




def process_single_image(path):

    imgpath = '1.jpg'

    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)

    processed_image = inception_preprocessing.preprocess_image(image,
                                                               image_size,
                                                               image_size,
                                                               is_training=False)

    processed_images = tf.expand_dims(processed_image, 0)
    #print(processed_images)
    #print(type(processed_images))
    #print(sess.run(processed_images))
    return processed_images

def main(_):
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        tf_global_step = slim.get_or_create_global_step()
        init = tf.initialize_all_variables()
        sess.run(init)
        ######################
        # Select the dataset #
        ######################
        dataset = AgriculturalDisease.get_split_test('test', '/media/zh/DATA/AgriculturalDisease20181023/tf_data')

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            num_epochs=1,
            common_queue_capacity=2 * 100,
            common_queue_min=100)

        [image, image_id] = provider.get(['image', 'image_id'])

        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(dataset.num_classes - FLAGS.labels_offset),
            is_training=False)
        preprocessing_name = FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        image_size = network_fn.default_image_size
        image = image_preprocessing_fn(image, image_size, image_size)

        images, image_ids = tf.train.batch(
            [image, image_id],
            batch_size=100,
            num_threads=4,
            capacity=5 * 100,
            allow_smaller_final_batch=True)

        #arg_scope = inception.inception_resnet_v2_arg_scope()
        #with slim.arg_scope(arg_scope):
        logits, end_points = network_fn(images)

        #checkpoint_file = '/media/zh/DATA/AgriculturalDisease/train_logs1/model.ckpt-40000'
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)

        predictions = tf.argmax(logits, 1)
        #labels = tf.squeeze(labels)

        num_batches = math.ceil(dataset.num_samples / float(100))
        tf.logging.info(
            'num_sample %d batch_size %d num_baatches %d' % (dataset.num_samples, 100, num_batches))
        probabilities = tf.nn.softmax(logits)
        #sess.run(tf.global_variables_initializer())
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)

        write_num = 0
        json_str = []
        for i in range(num_batches):

            class_id, ids = sess.run([predictions, image_ids])
            # write json
            for j in range(len(class_id)):
                write_num += 1
                tf.logging.info(
                    'write jsondata batch %d/%d  num  %d/%d' % (i, num_batches, write_num, dataset.num_samples))
                dic = {"disease_class": int(class_id[j]), "image_id": ids[j].decode()}
                #tf.logging.info(dic)
                json_str.append(dic)
        coord.request_stop()
        coord.join(threads)
        #print(json_str)
        filename = 'record_'+FLAGS.model_name+'.json'
        with open(filename, "w") as f:
            json.dump(json_str, f)
            print("write finish")

if __name__ == '__main__':
  tf.app.run()
