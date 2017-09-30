import tensorflow as tf
import configuration_params
import utils
import os
import read_from_records
from tensorflow.contrib.layers import fully_connected, conv2d, max_pool2d, dropout, batch_norm, optimize_loss
from tensorflow.contrib.framework import get_global_step
# from tflearn import conv_2d, max_pool_2d
# from tflearn import activations

# from tflearn import fully_connected
from tensorflow.contrib.losses import softmax_cross_entropy
from tensorflow.contrib.opt import LazyAdamOptimizer
from tensorflow.contrib.metrics import accuracy


class NatashaNet:
    # Build the graph for natasha model """
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

    def _create_inputs(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("inputs"):
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            self.example_batch, self.label_batch = read_from_records.get_new_data_batch(self.folder,
                                                                                        self.batch_size,
                                                                                        shuffle=shuffle)
            self.file_names = tf.convert_to_tensor(read_from_records.get_file_names(self.folder))

    def _create_logits(self):
        """ Step 3 : define the model """
        with tf.device('/gpu:0'):
            result = batch_norm(self.example_batch,
                                is_training=(self.folder == 'test'),
                                scope="batch_normalization_of_the_very_input")
            result = conv2d(result,
                            kernel_size=self.filter_size_on_1_layer,
                            num_outputs=self.number_of_filters_on_1_layer,
                            stride=1,
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            normalizer_fn=tf.contrib.layers.batch_norm,
                            normalizer_params={"is_training": (self.folder == 'test')},
                            scope="first_convolutional_layer")

            print('shape before the first pooling ', result.get_shape())
            result = max_pool2d(result,
                                kernel_size=self.pooling_scale,
                                stride=[2, 2],
                                padding='VALID',
                                scope="pooling_after_first_convolutional_layer")
            print('shape after the first pooling ', result.get_shape())

            result = conv2d(result,
                            kernel_size=self.filter_size_on_2_layer,
                            num_outputs=self.number_of_filters_on_2_layer,
                            stride=1,
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            normalizer_fn=tf.contrib.layers.batch_norm,
                            normalizer_params={"is_training": (self.folder == 'test')},
                            scope="second_convolutional_layer")

            result = max_pool2d(result,
                                kernel_size=self.pooling_scale,
                                stride=[2, 2],
                                padding='VALID',
                                scope="pooling_after_second_convolutional_layer")

            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            result = tf.reshape(result, [-1,
                                         result.get_shape().as_list()[1] *
                                         result.get_shape().as_list()[2] *
                                         result.get_shape().as_list()[3]])

            result = fully_connected(result,
                                     num_outputs=self.number_of_neurons_in_1_fully_connected_layer,
                                     scope='first_fully_connected_layer',
                                     activation_fn=tf.nn.elu,
                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                     normalizer_params={"is_training": (self.folder == 'test')})

            result = fully_connected(result,
                                     num_outputs=self.number_of_neurons_in_2_fully_connected_layer,
                                     scope='second_fully_connected_layer',
                                     activation_fn=tf.nn.elu,
                                     normalizer_fn=tf.contrib.layers.batch_norm,
                                     normalizer_params={"is_training": (self.folder == 'test')})

            # Apply Dropout
            result = dropout(result,
                             keep_prob=self.dropout,
                             scope="dropout")

            self.logits = fully_connected(result,
                                          num_outputs=configuration_params.num_labels,
                                          scope='model_output_logits',
                                          activation_fn=None,
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params={"is_training": (self.folder == 'test')})

        # a very useful summary to see if weights are updated or not
        # works only on cpu
        with tf.device('/cpu:0'):
            print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

            with tf.variable_scope('first_convolutional_layer', reuse=True):
                tf.summary.scalar('first_convolutional_layer/weights[3][3][1][10]',
                                  tf.get_variable('weights')[3][3][1][10])

    def _create_loss(self):
        """ Step 4: define the loss function """
        with tf.device('/cpu:0'):
            self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=self.logits,
                                                                       onehot_labels=self.label_batch,
                                                                       scope="loss"))
            tf.summary.scalar('self.loss', self.loss)

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = optimize_loss(loss=self.loss,
                                           global_step=self.global_step,
                                           learning_rate=configuration_params.learning_rate,
                                           optimizer='Adam',
                                           gradient_noise_scale=None,
                                           gradient_multipliers=None,
                                           clip_gradients=None,
                                           learning_rate_decay_fn=None,
                                           update_ops=None,
                                           variables=None,
                                           name='Adam_optimizer',
                                           summaries=[
                                               "learning_rate",
                                               "loss",
                                               "global_gradient_norm",
                                           ],
                                           colocate_gradients_with_ops=False,
                                           increment_global_step=True)
            tf.summary.scalar('global_step', self.global_step)

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
        # """ Build the graph for our model """
        self._create_inputs()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions_and_accuracy()

    def build_graph_with_output(self):
        # """ Build the graph for our model """
        self._create_inputs()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions_and_accuracy()
        self._output()
