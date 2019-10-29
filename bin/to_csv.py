import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', help='source .npy file', required=True)
    parser.add_argument('--dst', help='destination .csv file', required=True)
    parser.add_argument('--fmt', help='output file numbers format', default='%.18e')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    data = np.load(args.src)
    np.savetxt(args.dst, data, fmt=args.fmt, delimiter=", ")
