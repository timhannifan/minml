import argparse
import yaml
import click

from components.experiment import Experiment

class Minml():
    def __init__(self, config, in_path, out_path, test):
        self.config = config
        self.in_path = in_path
        self.out_path = out_path
        self.test = test

    def run(self):
        click.echo(f"Running Minml with config: {self.config}")
        with open(self.config) as f:
            loaded_config = yaml.load(f)

        exp = Experiment(
            arg_dict=loaded_config
        )
        exp.run()

if __name__ == "__main__":
    desc = ("Main method for running Minml pipeline")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', dest='config', type=str,
                        help=('Model config path'))
    parser.add_argument('--in_path', dest='in_path', type=str,
                        help=('Input data file location'))
    parser.add_argument('--out_path', dest='out_path', type=str, default='',
                        help=('Output directory'))
    parser.add_argument('--test', dest='test', type=bool, default=0,
                        help=('Select this option to run a test version'))
    args = parser.parse_args()
    args_dict = {'config': str(args.config),
                 'in_path': str(args.in_path),
                 'out_path': str(args.out_path),
                 'test': bool(args.test)}

    m = Minml(**args_dict)
    m.run()

