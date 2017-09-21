import os
import pathlib
import numpy as np
import tensorflow as tf
import datetime
import configuration_params
import net


def load_filenames(folder):
    f = open(configuration_params.labels_file, "r")
    lines = f.readlines()
    result = []
    list_of_files = os.listdir("./tfrecords/" + folder + "/")
    for x in lines:
        filename = x.split(' ')[0] + ".jpg.tfrecords"
        if filename in list_of_files:
            result.append(x.split(' ')[0])
    f.close()
    return result


def read_my_file_format(filename_queue):
    with tf.name_scope("reader"):
        reader = tf.TFRecordReader()

    with tf.name_scope("image"):
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'name': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)},
        name='name_for_operation_parse_single_example')

        image_raw = tf.cast(tf.decode_raw(features['image_raw'],
                                          tf.float64,
                                          name='name_for_image_raw'),
                            tf.float32)

        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        shape = tf.convert_to_tensor([height, width, configuration_params.num_channels])

        image = tf.reshape(image_raw, shape)

        resized_image = tf.image.resize_images(image, [configuration_params.unified_height,
                                                       configuration_params.unified_width])

    with tf.name_scope("labels"):
        # here we substruct 1 because our labels are 1, 2, 3 and one_hot works correctly only with labels like 0, 1, 2
        label = tf.cast(features['label'], tf.int32) - 1
        label_one_hot = tf.one_hot(label,
                                   on_value=1.0,
                                   off_value=0.0,
                                   depth=configuration_params.num_labels)

    return resized_image, label_one_hot


def input_pipeline(file_names, batch_size, num_epochs):
    filename_queue = tf.train.string_input_producer(file_names, num_epochs=num_epochs, shuffle=True)

    with tf.name_scope("inputs"):
        example, label = read_my_file_format(filename_queue)
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 1
        capacity = min_after_dequeue + 3 * batch_size

        example_batch, label_batch = tf.train.shuffle_batch([example, label],
                                                            batch_size=batch_size,
                                                            capacity=capacity,
                                                            num_threads=2,
                                                            min_after_dequeue=min_after_dequeue,
                                                            allow_smaller_final_batch=True)
        tf.summary.image('example_batch', example_batch)
    return example_batch, label_batch


def get_new_data_batch(folder):
    file_names = [("./tfrecords/" + folder + "/%s.jpg.tfrecords" % s) for s in load_filenames(folder)]
    print("file_names = ", file_names)
    return input_pipeline(file_names, configuration_params.batch_size, configuration_params.num_epoch)
