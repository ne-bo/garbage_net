import tensorflow as tf
import configuration_params
import utils
import os
import read_from_records


class NatashaNet:
    """ Build the graph for natasha model """

    def __init__(self, batch_size, learning_rate, folder):
        self.batch_size = batch_size
        self.lr = learning_rate
        self.folder = folder
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.dropout = 0.99
        self.pooling_scale = 2

    def _create_inputs(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("inputs"):
            if self.folder == 'test':
                shuffle = False
            else:
                shuffle = True
            self.example_batch, self.label_batch = read_from_records.get_new_data_batch(self.folder,
                                                                                        self.batch_size,
                                                                                        shuffle=shuffle)
            self.file_names = tf.convert_to_tensor(read_from_records.get_file_names(self.folder))

    # Create some wrappers for simplicity
    def _conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        with tf.name_scope("batch_normalization"):
            x = tf.contrib.layers.batch_norm(x,
                                             center=True, scale=True,
                                             is_training=(self.folder == 'test'))
        return tf.nn.relu(x)

    def _maxpool2d(self, x, k):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def _create_weights_and_biases(self):
        with tf.device('/gpu:0'):
            with tf.name_scope("weights"):
                filter_size_on_1_layer = 5
                number_of_filters_on_1_layer = 32

                filter_size_on_2_layer = 5
                number_of_filters_on_2_layer = 64

                number_of_neurons_in_1_fully_connected_layer = 1024

                height_after_1_pooling = configuration_params.unified_height // self.pooling_scale
                height_after_2_pooling = height_after_1_pooling // self.pooling_scale
                width_after_1_pooling = configuration_params.unified_width // self.pooling_scale
                width_after_2_pooling = width_after_1_pooling // self.pooling_scale

                self.weights = {
                    # conv 1
                    'wc1': tf.Variable(tf.truncated_normal([filter_size_on_1_layer,
                                                            filter_size_on_1_layer,
                                                            configuration_params.num_channels,
                                                            number_of_filters_on_1_layer])),
                    # conv 2
                    'wc2': tf.Variable(tf.truncated_normal([filter_size_on_2_layer,
                                                            filter_size_on_2_layer,
                                                            number_of_filters_on_1_layer,
                                                            number_of_filters_on_2_layer])),
                    # fully connected
                    'wd1': tf.Variable(tf.truncated_normal([height_after_2_pooling *
                                                            width_after_2_pooling *
                                                            number_of_filters_on_2_layer,
                                                            number_of_neurons_in_1_fully_connected_layer])),
                    # (class prediction)
                    'out': tf.Variable(tf.random_normal([number_of_neurons_in_1_fully_connected_layer,
                                                         configuration_params.num_labels]))
                }
        with tf.device('/gpu:0'):
            with tf.name_scope("biases"):
                self.biases = {
                    'bc1': tf.Variable(tf.truncated_normal([number_of_filters_on_1_layer])),
                    'bc2': tf.Variable(tf.truncated_normal([number_of_filters_on_2_layer])),
                    'bd1': tf.Variable(tf.truncated_normal([number_of_neurons_in_1_fully_connected_layer])),
                    'out': tf.Variable(tf.random_normal([configuration_params.num_labels]))
                }

    def _create_logits(self):
        """ Step 3 : define the model """
        with tf.device('/gpu:0'):
            with tf.name_scope("input"):
                # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
                x = self.example_batch
            with tf.name_scope("first_convolutional_layer"):
                # Convolution Layer
                conv1 = self._conv2d(x, self.weights['wc1'], self.biases['bc1'])

            with tf.name_scope("pooling_after_first_convolutional_layer"):
                # Max Pooling (down-sampling)
                conv1 = self._maxpool2d(conv1, k=self.pooling_scale)

            with tf.name_scope("second_convolutional_layer"):
                # Convolution Layer
                conv2 = self._conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
            with tf.name_scope("pooling_after_second_convolutional_layer"):
                # Max Pooling (down-sampling)
                conv2 = self._maxpool2d(conv2, k=self.pooling_scale)

            with tf.name_scope("fully_connected_layer"):
                # Fully connected layer
                # Reshape conv2 output to fit fully connected layer input
                fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
                fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
                fc1 = tf.nn.relu(fc1)

            with tf.name_scope("batch_normalization"):
                bn = tf.contrib.layers.batch_norm(fc1,
                                                  center=True, scale=True,
                                                  is_training=(self.folder == 'test'))

            with tf.name_scope("dropout"):
                # Apply Dropout
                bn = tf.nn.dropout(bn, self.dropout)

            with tf.name_scope("model_output_logits"):
                # Output, class prediction
                self.logits = tf.add(tf.matmul(bn, self.weights['out']), self.biases['out'])

    def _create_loss(self):
        """ Step 4: define the loss function """
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                # define loss function to be NCE loss function
                with tf.name_scope("loss"):
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                       labels=self.label_batch),
                                               name="loss")
                    tf.summary.scalar('loss', self.loss)

    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss,
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
        self._create_weights_and_biases()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions_and_accuracy()

    def build_graph_with_output(self):
        """ Build the graph for our model """
        self._create_inputs()
        self._create_weights_and_biases()
        self._create_logits()
        self._create_loss()
        self._create_optimizer()
        self._create_predictions_and_accuracy()
        self._output()
