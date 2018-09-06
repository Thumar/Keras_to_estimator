import os
from os import listdir
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img

bas_dir = "/home/manu/PycharmProjects/DataSet/cifar"

train_dir = join(bas_dir, "train")
test_dir = join(bas_dir, "test")

IMG_SIZE = 32
labels = {"airplane.png": 0,
          "automobile.png": 1,
          "bird.png": 2,
          "cat.png": 3,
          "deer.png": 4,
          "dog.png": 5,
          "frog.png": 6,
          "horse.png": 7,
          "ship.png": 8,
          "truck.png": 9}


def getFileList(dir):
    x = []
    y = []
    for f in listdir(dir):
        path = join(dir, f)
        if os.path.isfile(path):
            y.append(labels.get(f.split("_")[1]))
            x.append(path)
    return x, y


train_x, train_y = getFileList(train_dir)
test_x, test_y = getFileList(test_dir)

num_class = 10

train_y = keras.utils.to_categorical(train_y, num_class)
test_y = keras.utils.to_categorical(test_y, num_class)

print(len(train_x), len(train_y))
print(len(test_x), len(test_y))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(image_paths, labels, out_path):
    writer = tf.python_io.TFRecordWriter(out_path)

    for i in range(len(labels)):
        print(image_paths[i])
        im = np.array(img_to_array(load_img(image_paths[i], target_size=(IMG_SIZE, IMG_SIZE))) / 255.)
        g_labels = labels[i].astype(np.float32)
        example = tf.train.Example(features=tf.train.Features(
            feature={'image': _bytes_feature(im.tostring()),
                     'labels': _bytes_feature(
                         g_labels.tostring())
                     }))

        writer.write(example.SerializeToString())

    writer.close()


tfrecord_path = "/home/manu/PycharmProjects/Keras_estimator/data"

path_tfrecords_train = os.path.join(tfrecord_path, "train.tfrecords")
path_tfrecords_test = os.path.join(tfrecord_path, "test.tfrecords")

convert(train_x, train_y, path_tfrecords_train)
convert(test_x, test_y, path_tfrecords_test)
