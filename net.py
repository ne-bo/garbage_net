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

            # reshape image to 1-D tensor
            self.reshaped_example_batch = tf.reshape(self.example_batch,
                                                     [self.batch_size,
                                                      configuration_params.unified_width *
                                                      configuration_params.unified_height *
                                                      configuration_params.num_channels])

    def _create_logits(self):
        """ Step 3 : define the model """
        with tf.device('/cpu:0'):
            with tf.name_scope("weights_and_biases"):
                self.weights = tf.Variable(tf.truncated_normal([configuration_params.unified_height *
                                                                configuration_params.unified_width *
                                                                configuration_params.num_channels,
                                                                configuration_params.num_labels]), name="weights")
                self.biases = tf.Variable(tf.truncated_normal([configuration_params.num_labels]), name="biases")
            with tf.name_scope("model_output_logits"):
                self.logits = tf.nn.relu(tf.matmul(self.reshaped_example_batch, self.weights) + self.biases)

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
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss,
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
