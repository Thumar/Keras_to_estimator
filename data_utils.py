import os
from os import listdir
from os.path import join

import tensorflow as tf

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


# train_y = tf.keras.utils.to_categorical(train_y, num_class)
# test_y = tf.keras.utils.to_categorical(test_y, num_class)

def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = image_decoded / 255.0
    image_decoded.set_shape([32, 32, 3])
    return image_decoded


def input_fn(features, labels, num_epochs=None, shuffle=True, batch_size=32):
    img_filenames = tf.constant(features)
    img_labels = tf.constant(labels)

    dataset_x = tf.data.Dataset.from_tensor_slices(img_filenames)
    dataset_x = dataset_x.map(_parse_function)

    dataset_y = tf.data.Dataset.from_tensor_slices(img_labels)
    dataset_y = dataset_y.map(lambda z: tf.one_hot(z, num_class, dtype=tf.int32))

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


def train_input_fn(num_epoch, shuffle, batch_size):
    return input_fn(train_x, train_y, num_epoch, shuffle, batch_size)


def test_input_fn(num_epoch, shuffle, batch_size):
    return input_fn(test_x, test_y, num_epoch, shuffle, batch_size)


def dataset_input_fn(filenames, num_epochs):
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        featdef = {
            'image': tf.FixedLenFeature(shape=[], dtype=tf.string),
            'labels': tf.FixedLenFeature(shape=[], dtype=tf.string),
        }

        example = tf.parse_single_example(record, featdef)
        im = tf.decode_raw(example['image'], tf.float32)
        im = tf.reshape(im, (IMG_SIZE, IMG_SIZE, 3))
        lbl = tf.decode_raw(example['labels'], tf.float32)
        return im, lbl

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(128)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels


