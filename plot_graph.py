import sys
import argparse
import util

from pdb import set_trace as bp

def main(args):
    util.plot(args.result_file)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str,
        default='result/Threshold_color_FERET_fa_embeddings_v3')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))