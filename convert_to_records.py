import numpy as np
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import configuration_params


def load_array_with_map(filename):
    f = open(filename, "r")
    lines = f.readlines()
    result = np.ndarray([100000], dtype=int)
    for x in lines:
        result[int(x.split(' ')[0])] = int(x.split(' ')[1])
    f.close()
    return result


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(image, name, labels):
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]

    image = image - image.mean()

    print("rows, cols, depth = ", rows," ", cols," ", depth)
    print(image)
    int_name = int(name.replace('.jpg', ''))

    filename = os.path.join("./tfrecords/" + name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'name': _int64_feature(int_name),
            'label': _int64_feature(labels[int_name]),
            'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    writer.close()


def do_conversion_from_images_to_tfrecords():
    image_files = os.listdir(configuration_params.photos_directory)
    labels = load_array_with_map(configuration_params.labels_file)
    print(labels)

    for picture_number in image_files:
        try:
            print(datetime.datetime.now())
            image_data = plt.imread(configuration_params.photos_directory + picture_number).astype(int)
            convert_to_tfrecords(image_data, picture_number, labels)
        except IOError as e:
            print('Could not read:', picture_number, ':', e, '- it\'s ok, skipping.')

