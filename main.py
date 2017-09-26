import learning_process
import convert_to_records


def main():
    #convert_to_records.do_conversion_from_images_to_tfrecords() #we only need do conversion if new images are addede
    #learning_process.learning_on_the_training_set()
    learning_process.evaluation_on_the_test_set()




if __name__ == '__main__':
    main()
