from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import sys

import util
from database import Database
from pdb import set_trace as bp

def simulator(args, emb_array, labels, max_compare_num, filepath, threshold):
    # Initialize
    fa = 0  # False accept
    wa = 0  # Wrong answer
    fr = 0  # False reject
    accept = 0
    reject = 0

    # Construct database
    database = Database(emb_array.shape[0], max_compare_num)

    # Simulating
    for indx, emb in enumerate(emb_array):
        test_array, test_label = util.get_batch(emb_array, labels, indx)

        if len(database) != 0:  # train_array is not empty
            max_id, max_similarity = database.get_most_similar(test_array)
            # Not intruder
            if threshold < max_similarity:
                accept += 1
                if not database.contains(test_label):
                    fa += 1  # False accept
                elif test_label != database.get_label_by_id(max_id):
                    wa += 1  # Recognition error
            # Intruder
            else:
                reject += 1
                if database.contains(test_label):
                    fr += 1  # False reject

        # Add to database
        database.insert(test_label, test_array)

    #database.print_database()

    # Calculate error
    result_file = util.show_and_save_v3(fa, fr, wa, accept, reject, max_compare_num, filepath)
    return result_file

def main(args):
    filepaths = [args.csv_dir + v for v in os.listdir(args.csv_dir) if 'features' in v]
    for csv_filepath in filepaths:
        util.green_print(csv_filepath)
        filepath = util.create_output_path(csv_filepath)

        # Read embeddings
        emb_array, labels = util.readEmb_csv(csv_filepath)

        # Main
        #start = 0
        #step = 100
        #total_num = len(labels) + step
        #thds = np.arange(0, 4, 0.01)
        #for i in range(start, total_num, step):
        #for thd in thds:
        result_file = simulator(args, emb_array, labels, -1, filepath, args.threshold)

        # Plot
        #util.plot(result_file, start)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_dir', type=str, help='the csv filepath with features')
    parser.add_argument('threshold', type=float, help='the fixed threshold')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
