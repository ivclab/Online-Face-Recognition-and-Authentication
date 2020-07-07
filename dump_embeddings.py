"""
Use to dump facenet embeddings to csv file with the format of:
[name, feature, threshold(redundant), image_path]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import math
import csv
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf
import random

import facenet_simple

from pdb import set_trace as bp

def green_print(line):
    """Print in green"""
    print('\033[92m'+line+'\033[0m')

def my_get_paths(data_dir, ext):
    """Get filepath from data_dir"""
    path_list = []
    for p in os.listdir(data_dir):
        class_folder = os.path.join(data_dir, p)
        if os.path.isdir(class_folder):
            for pp in os.listdir(class_folder):
                file_path = os.path.join(class_folder, pp)
                if os.path.exists(file_path) and pp.endswith(ext):
                    path_list.append(file_path)
    green_print('Get %d path' % len(path_list))
    return path_list

def generate_register_order(emb_paths, filename, exp_time):
    filename = 'data/register_order_' + filename.split('.')[0] + '_v' + exp_time + '.txt'

    # Do not generate again if exists
    if os.path.exists(filename):
        green_print('%s exists' % filename)
        with open(filename, 'r') as textfile:
            register_order = textfile.read().split('\n')
        return register_order

    register_order = []

    '''
    # Get class
    classes = [v.split('/')[-2] for v in emb_paths]
    class_set = list(set(classes))

    # Random choose the first emedding of a class to register
    for c in class_set:
        cur_class_paths = [v for v in emb_paths if c == v.split('/')[-2]]
        first  = random.choice(cur_class_paths)
        register_order += [first]
        emb_paths.remove(first)
    '''

    # Shuffle the rest
    random.shuffle(emb_paths)
    register_order = emb_paths

    # write to txt
    with open(filename, "w") as txt_file:
        for order in register_order[:-1]:
            txt_file.write(order + '\n')
        txt_file.write(register_order[-1])
    return register_order


def write_to_csv(filename, emb_paths, emb_array, exp_time, register_order_file=None):
    """Write to csv file in the format of: [name, features, threshold, path]
    Args:
        filename: The filename of output csv file
        emb_paths: The image paths of the embeddings
        emb_array: The embeddings generated from FaceNet,
                same order with emb_paths
        register_order_file:The order of the images registered, 
                if none, will be the order of the emb_paths.
    """
    # Construct path and emb dict
    keys = [v.split('/')[-2]+'/'+v.split('/')[-1] for v in emb_paths]
    emb_dict = dict(zip(keys, emb_array))

    # Read register order
    if register_order_file is None:
        register_order = keys
    elif register_order_file == 'auto-gen':
        register_order = generate_register_order(keys, filename, exp_time)
    else:
        with open(register_order_file, 'r') as textfile:
            register_order = textfile.read().split('\n')

    filename = 'data/features_' + filename + '_v' + exp_time + '.csv'
    with open(filename, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        for path in register_order:
            name = path.split('/')[0]
            features = emb_dict[path]
            threshold = 0  # Redundant
            csv_writer.writerow([name, features, threshold, path])
    csv_file.close()
    green_print("Finish Write the CSV file: %s " %(filename))


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        with tf.Session(config=config) as sess:

            # Get the paths for the corresponding images
            data_paths = my_get_paths(args.data_dir, args.image_ext)

            # Load the model
            facenet_simple.load_model(args.model)
            green_print('Model loaded')

            # Get input and output tensors
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            green_print('Runnning forward pass on %d images' % len(data_paths))
            nrof_images = len(data_paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = data_paths[start_index:end_index]
                images = facenet_simple.load_data(paths_batch, args.image_size)
                feed_dict = {images_placeholder:images, phase_train_placeholder:False}
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    	    # Write to csv
            filename = args.csv_filename
            if args.csv_filename is None:
                filename = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
            
	    # Write to file
            for i in range(10):
                write_to_csv(filename, data_paths, emb_array, str(i), args.register_order_file)

    green_print('END')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='model used to dump embeddings',
                        default='20170512-110547/')
    parser.add_argument('data_dir', type=str, help='images root folder')
    parser.add_argument('--register_order_file', type=str, help='the register order file',
                        default='auto-gen')

    parser.add_argument('--csv_filename', type=str, help='filename of output csv file')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=160)
    parser.add_argument('--image_ext', type=str, default='png')

    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8)
    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
