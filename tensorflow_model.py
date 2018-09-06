import tensorflow as tf


def cnn_model():
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    use_bias = False

    # Layer 1
    conv = tf.keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(input_layer)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    conv = tf.keras.layers.Conv2D(32,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(activation)
    dropout = tf.keras.layers.Dropout(0.2)(max_pool)

    # Layer 2
    conv = tf.keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(dropout)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    conv = tf.keras.layers.Conv2D(64,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.3)(max_pool)

    # Layer 3
    conv = tf.keras.layers.Conv2D(128,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(dropout)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    conv = tf.keras.layers.Conv2D(128,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  use_bias=use_bias,
                                  activation=None)(activation)
    bn = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(conv)
    activation = tf.keras.layers.Activation('relu')(bn)

    max_pool = tf.keras.layers.MaxPooling2D()(activation)
    dropout = tf.keras.layers.Dropout(0.4)(max_pool)

    flatten = tf.keras.layers.Flatten()(dropout)

    # Output layers: separate outputs for the weather and the ground labels
    output = tf.keras.layers.Dense(10,
                                   activation='softmax',
                                   name='output')(flatten)

    return tf.keras.Model(inputs=input_layer, outputs=output)
