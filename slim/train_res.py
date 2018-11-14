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




def read_tf_record():
    return 1
#»ñÈ¡inception_resnet_v2Ä¬ÈÏÍ¼Æ¬³ß´ç£¬ÕâÀïÎª299
image_size = inception.inception_resnet_v2.default_image_size
#»ñÈ¡imagenetËùÓÐ·ÖÀàµÄÃû×Ö£¬ÕâÀïÓÐ1000¸ö·ÖÀà
#names = imagenet.create_readable_names_for_imagenet_labels()

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

with tf.Session() as sess:
    tf.logging.set_verbosity(tf.logging.INFO)
    tf_global_step = slim.get_or_create_global_step()
    init = tf.initialize_all_variables()
    sess.run(init)
    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        'AgriculturalDisease', 'validation', '/media/zh/DATA/AgriculturalDisease/tf_data')

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        num_epochs=1,
        common_queue_capacity=2 * 100,
        common_queue_min=100)

    [image, label] = provider.get(['image', 'label'])
    label -= 0

    image_size = inception.inception_resnet_v2.default_image_size
    image = inception_preprocessing.preprocess_image(image,
                                             image_size,
                                             image_size,
                                             is_training=False)
    images, labels = tf.train.batch(
        [image, label],
        batch_size=50,
        num_threads=4,
        capacity=5 * 50,
        allow_smaller_final_batch=True)

    arg_scope = inception.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception.inception_resnet_v2(images, num_classes=61, is_training=False)

    checkpoint_file = '/media/zh/DATA/AgriculturalDisease/train_logs/model.ckpt-14552'
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    num_batches = math.ceil(dataset.num_samples / float(50))
    tf.logging.info(
        'num_sample %d batch_size %d num_baatches %d' % (dataset.num_samples, 50, num_batches))
    probabilities = tf.nn.softmax(logits)
    with tf.name_scope('eval'):
        eval = tf.nn.in_top_k(probabilities, labels, k=1)
    #sess.run(tf.global_variables_initializer())
    tf.local_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    acc_num = 0
    num = 0
    for i in range(num_batches):
        print(i)
        acc, tr = sess.run([eval, labels])
        acc_num += np.sum(acc)
        num += len(tr)
        print(np.sum(acc), tr)
    coord.request_stop()
    coord.join(threads)
    print(acc_num/dataset.num_samples)
    print(num)
