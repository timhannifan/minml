import argparse

class Minml():
    def __init__(self, in_path, out_path, test, verbose):
        pass

    def run(self, config_path):
        pass

if __name__ == "__main__":

    desc = ("Main method for running Minml pipeline")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--in_path', dest='in_path', type=str,
                        help=('Input data file location'))
    parser.add_argument('--out_path', dest='out_path', type=str, default='',
                        help=('Select this option to run a test version'))
    parser.add_argument('--test', dest='test', type=bool, default=0,
                        help=('Select this option to run a test version'))
    parser.add_argument('--verbose', dest='verbose', type=bool, default=0,
                        help=("Select verbose logging option."))
    args = parser.parse_args()
    args_dict = {
                 'in_path': str(args.in_path),
                 'out_path': str(args.out_path),
                 'test': bool(args.test),
                 'verbose': bool(args.verbose)}

    # print(**args_dict)
    m = Minml(**args_dict)

