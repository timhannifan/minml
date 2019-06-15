import argparse
import yaml
import click

from components.experiment.basic import Singlethread

class Minml():
    def __init__(self, config, in_path, out_path, test, verbose):
        self.config = config
        self.in_path = in_path
        self.out_path = out_path
        self.test = test
        self.verbose = verbose

    def run(self):
        click.echo(f"Running Minml with config: {self.config}")
        with open(self.config) as f:
            loaded_config = yaml.load(f)

        experiment = Singlethread(
            config=loaded_config
        )
        experiment.run()

if __name__ == "__main__":
    desc = ("Main method for running Minml pipeline")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', dest='config', type=str,
                        help=('Model config path'))
    parser.add_argument('--in_path', dest='in_path', type=str,
                        help=('Input data file location'))
    parser.add_argument('--out_path', dest='out_path', type=str, default='',
                        help=('Select this option to run a test version'))
    parser.add_argument('--test', dest='test', type=bool, default=0,
                        help=('Select this option to run a test version'))
    parser.add_argument('--verbose', dest='verbose', type=bool, default=0,
                        help=("Select verbose logging option."))
    args = parser.parse_args()
    args_dict = {'config': str(args.config),
                 'in_path': str(args.in_path),
                 'out_path': str(args.out_path),
                 'test': bool(args.test),
                 'verbose': bool(args.verbose)}

    m = Minml(**args_dict)
    m.run()

