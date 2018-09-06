import os
from functools import partial
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

import data_utils
import tensorflow_model

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

num_classes = 10


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

class_maping = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def getFileList(dir):
    x = []
    y = []
    for f in os.listdir(dir):
        path = join(dir, f)
        if os.path.isfile(path):
            y.append(labels.get(f.split("_")[1]))
            x.append(path)
    return x, y


def prediction_data():
    bas_dir = "/home/manu/PycharmProjects/DataSet/cifar"

    predict_dir = join(bas_dir, "predict")

    predict_image, true_labels = getFileList(predict_dir)

    return predict_image[1:45], true_labels[1:45]


def _parse_function(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = image_decoded
    image_decoded.set_shape([32, 32, 3])
    return {"input_1": image_decoded}


def predict_input_fn(image_path):
    img_filenames = tf.constant(image_path)

    dataset = tf.data.Dataset.from_tensor_slices(img_filenames)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    return image





model = tensorflow_model.cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

cifar_est = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir="kkt")


tfrecord_path = "/home/manu/PycharmProjects/Keras_estimator/data"

train_data = os.path.join(tfrecord_path, "train.tfrecords")
train_input = lambda: data_utils.dataset_input_fn(train_data, None)
cifar_est.train(input_fn=train_input, steps=7000)

test_data = os.path.join(tfrecord_path, "test.tfrecords")
test_input = lambda: data_utils.dataset_input_fn(test_data, 1)
res = cifar_est.evaluate(input_fn=test_input, steps=1)

print(res)

model_input_name = model.input_names[0]


def serving_input_receiver_fn():
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    images = tf.map_fn(partial(tf.image.decode_image, channels=1), input_ph, dtype=tf.uint8)
    images = tf.cast(images, tf.float32) / 255.
    images.set_shape([None, 32, 32, 3])

    return tf.estimator.export.ServingInputReceiver({model_input_name: images}, {'bytes': input_ph})


export_path = cifar_est.export_savedmodel('./export', serving_input_receiver_fn=serving_input_receiver_fn)

import matplotlib.image as mpimg

predict_image, true_label = prediction_data()

predict_result = list(cifar_est.predict(input_fn=lambda: predict_input_fn(predict_image)))

pos = 1
for img, lbl, predict_lbl in zip(predict_image, true_label, predict_result):
    output = np.argmax(predict_lbl.get('output'), axis=None)
    plt.subplot(4, 11, pos)
    img = mpimg.imread(img)
    plt.imshow(img)
    plt.axis('off')
    if output == lbl:
        plt.title(class_maping[output])
    else:
        plt.title(class_maping[output] + "/" + class_maping[lbl], color='#ff0000')
    pos += 1

plt.show()
