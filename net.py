import tensorflow as tf
import configuration_params
import utils
import os
import read_from_records
from tensorflow.contrib.layers import fully_connected, conv2d, max_pool2d, dropout
# from tflearn import conv_2d, max_pool_2d
# from tflearn import activations

# from tflearn import fully_connected
from tensorflow.contrib.losses import softmax_cross_entropy
from tensorflow.contrib.opt import LazyAdamOptimizer
from tensorflow.contrib.metrics import accuracy


class NatashaNet:
    """ Build the graph for natasha model """

    def __init__(self, batch_size, learning_rate, folder):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.folder = folder
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        self.dropout = 1.0
        self.pooling_scale = 2

        self.filter_size_on_1_layer = 5
        self.number_of_filters_on_1_layer = 20

        self.filter_size_on_2_layer = 3
        self.number_of_filters_on_2_layer = 20

        self.number_of_neurons_in_1_fully_connected_layer = 64
        self.number_of_neurons_in_2_fully_connected_layer = 32

        self.height_after_1_pooling = configuration_params.unified_height // self.pooling_scale
        self.height_after_2_pooling = self.height_after_1_pooling // self.pooling_scale
        self.width_after_1_pooling = configuration_params.unified_width // self.pooling_scale
        self.width_after_2_pooling = self.width_after_1_pooling // self.pooling_scale

    def _create_inputs(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("inputs"):
            if self.folder == 'test':
                shuffle = False
            else:
                shuffle = True
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            self.example_batch, self.label_batch = read_from_records.get_new_data_batch(self.folder,
                                                                                        self.batch_size,
                                                                                        shuffle=shuffle)
            self.file_names = tf.convert_to_tensor(read_from_records.get_file_names(self.folder))

    def _create_logits(self):
        """ Step 3 : define the model """
        with tf.device('/cpu:0'):

            conv1 = conv2d(self.example_batch,
                           kernel_size=self.filter_size_on_1_layer,
                           num_outputs=self.number_of_filters_on_1_layer,
                           stride=1,
                           padding='SAME',
                           activation_fn=tf.nn.elu,
                           normalizer_fn=tf.contrib.layers.batch_norm,
                           normalizer_params={"is_training": (self.folder == 'test')},
                           scope="first_convolutional_layer")

            conv1 = max_pool2d(conv1,
                               kernel_size=self.pooling_scale,
                               stride=1,
                               padding='SAME',
                               scope="pooling_after_first_convolutional_layer")

            conv2 = conv2d(conv1,
                           kernel_size=self.filter_size_on_2_layer,
                           num_outputs=self.number_of_filters_on_2_layer,
                           stride=1,
                           padding='SAME',
                           activation_fn=tf.nn.elu,
                           normalizer_fn=tf.contrib.layers.batch_norm,
                           normalizer_params={"is_training": (self.folder == 'test')},
                           scope="second_convolutional_layer")

            conv2 = max_pool2d(conv2,
                               kernel_size=self.pooling_scale,
                               stride=1,
                               padding='SAME',
                               scope="pooling_after_second_convolutional_layer")

            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1,
                                     conv2.get_shape().as_list()[1] *
                                     conv2.get_shape().as_list()[2] *
                                     conv2.get_shape().as_list()[3] ])

            fc1 = fully_connected(fc1,
                                   num_outputs=self.number_of_neurons_in_1_fully_connected_layer,
                                   scope='first_fully_connected_layer',
                                   activation_fn=tf.nn.elu,
                                   normalizer_fn=tf.contrib.layers.batch_norm,
                                   normalizer_params={"is_training": (self.folder == 'test')})

            fc2 = fully_connected(fc1,
                                   num_outputs=self.number_of_neurons_in_2_fully_connected_layer,
                                   scope='second_fully_connected_layer',
                                   activation_fn=tf.nn.elu,
                                   normalizer_fn=tf.contrib.layers.batch_norm,
                                   normalizer_params={"is_training": (self.folder == 'test')})

            # Apply Dropout
            do = dropout(fc2,
                         keep_prob=self.dropout,
                         scope="dropout")

            self.logits = fully_connected(do,
                                          num_outputs=configuration_params.num_labels,
                                          scope='model_output_logits',
                                          activation_fn=tf.nn.softmax,
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params={"is_training": (self.folder == 'test')})

    def _create_loss(self):
        """ Step 4: define the loss function """
        with tf.device('/cpu:0'):
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.logits,
                                                                       onehot_labels=self.label_batch,
                                                                       scope="loss"))
            tf.summary.scalar('loss', self.loss)

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = LazyAdamOptimizer(learning_rate=self.lr,
                                               name="LazyAdam").minimize(self.loss,
                                                                         global_step=self.global_step)

    def _create_predictions_and_accuracy(self):
        with tf.device('/cpu:0'):
            # Predictions
            with tf.name_scope(self.folder + "_prediction"):
                self.prediction = tf.nn.softmax(self.logits, name=self.folder + "_prediction")
            with tf.name_scope(self.folder + "_accuracy"):
                correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label_batch, 1))
                # Calculate accuracy
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar(self.folder + "_accuracy", self.accuracy)

    def _output(self):
        with tf.device('/cpu:0'):
            with tf.name_scope(self.folder + "_output"):
                self.output_labels = tf.argmax(self.prediction, axis=1) + 1

    def build_graph(self):
        """ Build the graph for our model """
        self._create_inputs()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions_and_accuracy()

    def build_graph_with_output(self):
        """ Build the graph for our model """
        self._create_inputs()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions_and_accuracy()
        self._output()
