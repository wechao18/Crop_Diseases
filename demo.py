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
from preprocessing import preprocessing_factory
from nets import nets_factory
import matplotlib as mpl




slim = tf.contrib.slim


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



def process_single_image(path):


    image = tf.image.decode_jpeg(tf.read_file(path), channels=3)


    print(image.shape)


    processed_image = inception_preprocessing.preprocess_image(image,
                                                               image_size,
                                                               image_size,
                                                               is_training=False)
    with tf.Session() as sess:
        img = sess.run(processed_image)
        img1 = sess.run(image)
        plt.figure()
        plt.imshow(img1)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
    #print(processed_images)
    processed_images = tf.expand_dims(processed_image, 0)
    print(processed_images.shape)

    return processed_images

def vis_square(data, padsize=1, padval=0):
    data = (data - data.min()) / (data.max() - data.min()) * 255
    print(data.shape)
    print(data)
    # 让合成图为方
    n = int(np.ceil(np.sqrt(data.shape[0])))
    print(n,data.ndim)
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    print(padding)
    data = np.pad(data, padding, mode='constant', constant_values=(255, 255))  # pad with ones (white)
    # 将所有输入的data图像平复在一个ndarray-data中（tile the filters into an image）
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    # data的一个小例子,e.g., (3,120,120)
    # 即，这里的data是一个2d 或者 3d 的ndarray
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print(data.shape, data)
    # 显示data所对应的图像
    cmap = plt.cm.get_cmap('jet') #gist_heat
    #cm = plt.cm.get_cmap('binary')
    def colormap():
        return mpl.colors.LinearSegmentedColormap.from_list('cmap',
                                                            ['#00117C', '#0024EE', '#00FFC1', '#830005', '#FFFF00', '#FF0000',
                                                             '#FFFFFF'], 256)

    plt.imshow(data, cmap=colormap())
    plt.axis('off')

    plt.show()


def main(_):

    path = '/media/zh/DATA/AgriculturalDisease/AgriculturalDisease_trainingset/images/0a7af6af6dd7a77b80f1e0c83a3c0e49.jpg'
    with tf.Session() as sess:
        image = tf.image.decode_jpeg(tf.read_file(path), channels=3)
        tf.logging.set_verbosity(tf.logging.INFO)
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(61 - FLAGS.labels_offset),
            is_training=False)
        preprocessing_name = FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        image_size = network_fn.default_image_size
        image = image_preprocessing_fn(image, image_size, image_size)
        image = tf.expand_dims(image, 0)

        logits, _, conv1, conv2, conv3, conv4 ,conv5, conv6, conv7 = network_fn(image)

        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.checkpoint_path)

        predictions = tf.argmax(logits, 1)


        tf.logging.info('111')
        probabilities = tf.nn.softmax(logits)
        #with tf.name_scope('eval'):
        #    eval1 = tf.nn.in_top_k(probabilities, labels, k=1)
        #    eval5 = tf.nn.in_top_k(probabilities, labels, k=5)

        tf.local_variables_initializer().run()

        pro, conv_view1, conv_view2, conv_view3, conv_view4,conv_view5, conv_view6, conv_view7= sess.run([probabilities, conv1, conv2, conv3, conv4,conv5, conv6, conv7])
        convview_reshape1 = sess.run(tf.transpose(conv_view1[0], [2, 0, 1, ]))
        convview_reshape2 = sess.run(tf.transpose(conv_view2[0], [2, 0, 1, ]))
        convview_reshape3 = sess.run(tf.transpose(conv_view3[0], [2, 0, 1, ]))
        convview_reshape4 = sess.run(tf.transpose(conv_view4[0], [2, 0, 1, ]))
        convview_reshape5 = sess.run(tf.transpose(conv_view5[0], [2, 0, 1, ]))
        convview_reshape6 = sess.run(tf.transpose(conv_view6[0], [2, 0, 1, ]))
        convview_reshape7 = sess.run(tf.transpose(conv_view7[0], [2, 0, 1, ]))
        #print(pro, conv_view.shape)
        vis_square(convview_reshape1)
        vis_square(convview_reshape2)
        vis_square(convview_reshape3)
        vis_square(convview_reshape4)
        vis_square(convview_reshape5)
        #vis_square(convview_reshape6)
        #vis_square(convview_reshape7)

    #process_single_image('/media/dell/D/dell/Disease/AgriculturalDisease_testA/images/1da2cd46-6067-4b47-a453-e02754962329___UF.GRC_YLCV_Lab 01728.JPG')
if __name__ == '__main__':
    tf.app.run()
