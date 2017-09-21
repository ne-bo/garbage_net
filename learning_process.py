import os
import pathlib
import numpy as np
import tensorflow as tf
import datetime
import configuration_params
import net
import read_from_records
import utils


def learning_on_the_training_set():

    model = net.NatashaNet(configuration_params.batch_size,
                           learning_rate=configuration_params.learning_rate,
                           folder = 'train')
    model.build_graph()

    # Predictions for the training
    with tf.name_scope("train_prediction"):
        train_prediction = tf.nn.softmax(model.logits, name="train_prediction")
    with tf.name_scope("train_accuracy"):
        correct_prediction = tf.equal(tf.argmax(train_prediction, 1), tf.argmax(model.label_batch, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('train_accuracy', accuracy)

    # ******************************************************************************************************************
    #
    # Running of the graph for training
    #
    # ******************************************************************************************************************

    utils.make_dir('checkpoints')
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # The op for initializing the variables.
        with tf.name_scope("init"):
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            sess.run(init_op)

        # Merge all the summaries and write them out to current dir
        merged = tf.summary.merge_all()
        writer_graph = tf.summary.FileWriter('.', sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        i = 0
        try:
            while not (coord.should_stop() or i == configuration_params.num_epoch):
                summary, o,l, tp = sess.run([merged,
                                       model.optimizer,
                                       model.loss,
                                       train_prediction
                ])

                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                # if that checkpoint exists, restore from checkpoint
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)

                if i % 10 == 0:
                    writer_graph.add_summary(summary, i)
                    print(datetime.datetime.now())
                    print('i = ', i)
                    saver.save(sess, 'checkpoints/natasha-model', i)

                i = i + 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            writer_graph.close()
        # Wait for threads to finish.
        coord.join(threads)

        sess.close()


def evaluation_on_the_test_set():
    """Eval loop."""
    model = net.NatashaNet(configuration_params.test_batch_size,
                           learning_rate=configuration_params.learning_rate,
                           folder = 'test')
    model.build_graph()
    print("test_batch_size = ", configuration_params.test_batch_size)
    # Predictions for the training
    with tf.name_scope("test_prediction"):
        test_prediction = tf.nn.softmax(model.logits, name="test_prediction")
    with tf.name_scope("test_accuracy"):
        correct_prediction = tf.equal(tf.argmax(test_prediction, 1), tf.argmax(model.label_batch, 1))
        # Calculate accuracy
        test_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('test_accuracy', test_accuracy)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # The op for initializing the variables.
        with tf.name_scope("init_test"):
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

            sess.run(init_op)

        # Merge all the summaries and write them out to current dir
        merged = tf.summary.merge_all()
        writer_graph = tf.summary.FileWriter('.', sess.graph)

        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        i = 0
        try:
            while not (coord.should_stop() or i == 2):
                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                # if that checkpoint exists, restore from checkpoint
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(ckpt.model_checkpoint_path)

                summary, prediction, accuracy, b, w = sess.run([merged, test_prediction, test_accuracy, model.biases, model.weights])
                writer_graph.add_summary(summary)
                print('i = ', i)
                print('test_predictions = ', prediction)
                print('test_accuracy = ', accuracy)
                print('model_biases = ', b)
                print('model_weights = ', w)

                i = i + 1

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')

        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            writer_graph.close()

        # Wait for threads to finish.
        coord.join(threads)

        sess.close()
