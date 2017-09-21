import numpy as np
import os
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
import configuration_params
from sklearn import model_selection


def load_array_with_map(filename):
    f = open(filename, "r")
    lines = f.readlines()
    result_lables = []
    result_images = []
    for x in lines:
        result_images.append((x.split(' ')[0]) + ".jpg")
        result_lables.append(int(x.split(' ')[1]))
    f.close()
    return result_images, result_lables


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to_tfrecords(image, name, label, folder):
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]

    image = image - image.mean()

    print("rows, cols, depth = ", rows," ", cols," ", depth)
    #print(image)
    int_name = int(name.replace('.jpg', ''))

    filename = os.path.join("./tfrecords/" + folder + "/" + name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    image_raw = image.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'name': _int64_feature(int_name),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    writer.close()


def do_conversion_from_images_to_tfrecords():
    #image_files = os.listdir(configuration_params.photos_directory)
    image_files, labels = load_array_with_map(configuration_params.labels_file)

    print(len(labels))
    print(labels)

    image_files_train, image_files_test, labels_train, labels_test = model_selection.train_test_split(image_files,
                                                                                                      labels,
                                                                                                      random_state=42,
                                                                                                      stratify=labels,
                                                                                                      test_size=0.2)
    for i in range(len(image_files_train)):
        try:
            print(datetime.datetime.now())
            picture_name = image_files_train[i]
            label = labels_train[i]
            image_data = plt.imread(configuration_params.photos_directory + picture_name).astype(int)

            print(picture_name, label)
            convert_to_tfrecords(image_data, picture_name, label, 'train')
        except IOError as e:
            print('Could not read:', picture_name, ':', e, '- it\'s ok, skipping.')

    for i in range(len(image_files_test)):
        try:
            print(datetime.datetime.now())
            picture_name = image_files_test[i]
            label = labels_test[i]
            image_data = plt.imread(configuration_params.photos_directory + picture_name).astype(int)

            convert_to_tfrecords(image_data, picture_name, label, 'test')
        except IOError as e:
            print('Could not read:', picture_name, ':', e, '- it\'s ok, skipping.')
