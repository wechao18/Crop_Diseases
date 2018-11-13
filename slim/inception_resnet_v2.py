# encoding:utf-8
import tensorflow as tf

from matplotlib import pyplot as plt
from nets import inception
from preprocessing import inception_preprocessing
import numpy as np
from datasets import imagenet
import json
import os



def read_tf_record():
    return 1
#»ñÈ¡inception_resnet_v2Ä¬ÈÏÍ¼Æ¬³ß´ç£¬ÕâÀïÎª299
image_size = inception.inception_resnet_v2.default_image_size
#»ñÈ¡imagenetËùÓÐ·ÖÀàµÄÃû×Ö£¬ÕâÀïÓÐ1000¸ö·ÖÀà
names = imagenet.create_readable_names_for_imagenet_labels()

slim = tf.contrib.slim


batch_size = 1
image_test = tf.placeholder(tf.float32, [batch_size, 299, 299, 3])
label_test = tf.placeholder(tf.int32, [batch_size])




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



    arg_scope = inception.inception_resnet_v2_arg_scope()
    with slim.arg_scope(arg_scope):
        logits, end_points = inception.inception_resnet_v2(image_test, num_classes=61, is_training=False)

    checkpoint_file = '/media/zh/DATA/AgriculturalDisease/train_logs/model.ckpt-14552'

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)


    probabilities = tf.nn.softmax(logits)
    with tf.name_scope('eval'):
        eval = tf.nn.in_top_k(probabilities, label_test, k=1)
    # detect images
    #files = tf.train.match_filenames_once('/media/zh/DATA/AgriculturalDisease/tf_data/AgriculturalDisease_validation_*.tfrecord')
    #img, label = read_tf_record(files)
    with open('/media/zh/DATA/AgriculturalDisease/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json', encoding='utf-8') as f:
        json_data = json.load(f)
        st = np.zeros(61)
        class_num = np.zeros(61)
        t_num = 0
        for i in range(len(json_data)):
            filename = os.path.join('/media/zh/DATA/AgriculturalDisease/AgriculturalDisease_validationset/images', json_data[i]['image_id'])
            processed_images = process_single_image(filename)
            label = tf.expand_dims(json_data[i]['disease_class'], 0)
            logit_values, acc = sess.run([probabilities, eval], feed_dict={image_test: sess.run(processed_images), label_test:sess.run(label)})
            #print(i,np.max(logit_values))
            print(i, np.argmax(logit_values), json_data[i]['disease_class'])
            class_num[json_data[i]['disease_class']] += 1
            st[np.argmax(logit_values)] += 1
            #print(np.sum(acc))
            t_num += np.sum(acc)

        print(class_num)
        print(st)
        print('acc: %d' %(t_num/len(json_data)))