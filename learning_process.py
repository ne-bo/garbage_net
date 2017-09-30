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
                           folder='train')
    model.build_graph()

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

        try:
            i = model.global_step.eval()
            while not (coord.should_stop() or i == configuration_params.num_epoch):
                #model._create_inputs()
                summary, o, l, tp, gs = sess.run([merged,
                                                   model.loss,
                                                   model.optimizer,
                                                   model.prediction,
                                                   model.global_step
                                                   ])


                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                # if that checkpoint exists, restore from checkpoint
                #if ckpt and ckpt.model_checkpoint_path:
                #    saver.restore(sess, ckpt.model_checkpoint_path)

                if i % 10 == 0:
                    writer_graph.add_summary(summary, global_step=i)
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
                           folder='test')
    model.build_graph_with_output()
    print("test_batch_size = ", configuration_params.test_batch_size)

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
            while not (coord.should_stop() or i == 1):
                ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/checkpoint'))
                # if that checkpoint exists, restore from checkpoint
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    print(ckpt.model_checkpoint_path)

                summary, accuracy, predicted_labels, files = sess.run([merged,
                                                                       model.accuracy,
                                                                       model.output_labels,
                                                                       model.file_names])
                writer_graph.add_summary(summary)
                print('i = ', i)
                print('test_accuracy = ', accuracy)
                for j in range(predicted_labels.shape[0]):
                    print(files[j], " ", predicted_labels[j])
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
