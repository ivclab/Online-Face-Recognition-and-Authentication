from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from pdb import set_trace as bp


def main(args):
    accuracies = 0.0
    for i in range(10):
        filepath = args.result_file + str(i)
        with open(filepath, 'r') as textfile:
            all_file = textfile.read().split('\n')
            all_file = [v for v in all_file if 'acc' in v]
        acc = [v.split('acc:')[1] for v in all_file]
        acc = [v.split('(')[0] for v in acc]
        acc = [float(v) for v in acc]
        max_acc = max(acc)
        #thd = acc.index(max_acc)
        print('v%d, %f' % (i, max_acc))
        accuracies += max_acc
    print('%f' % (accuracies/10))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', type=str, help='the result filepath')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))