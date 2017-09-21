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

    model = net.NatashaNet(configuration_params.batch_size, learning_rate=0.5, folder = 'train')
    model.build_graph()

    # Predictions for the training
    with tf.name_scope("train_prediction"):
        train_prediction = tf.nn.softmax(model.logits, name="train_prediction")
    with tf.name_scope("train_accuracy"):
        train_accuracy = tf.metrics.accuracy(model.label_batch, train_prediction, name="train_accuracy")
        tf.summary.scalar('train_accuracy', train_accuracy[0])

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
