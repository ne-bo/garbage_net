import os

dataset_directory = "/home/natasha/PycharmProjects/garbage_dataset/"
photos_directory = dataset_directory + "photos/"
labels_file = dataset_directory + "labels"
directory_for_saving_results = "./results/"

learning_rate = 0.0001
batch_size = 128
num_labels = 2
num_epoch = 3000

num_channels = 3

unified_height = 64
unified_width = 64

skip_step = 10


test_batch_size = len(os.listdir("./tfrecords/test/"))
full_train_batch_size = len(os.listdir("./tfrecords/train/"))