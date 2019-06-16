import argparse
import yaml
import click

from components.experiment import Experiment

class Minml():
    def __init__(self, config, load_db):
        self.config = config
        self.load_db = load_db

    def run(self):
        click.echo(f"Running Minml with config: {self.config}")
        with open(self.config) as f:
            loaded_config = yaml.load(f)

        exp = Experiment(
            arg_dict=loaded_config,
            load_db=self.load_db)
        exp.run()

if __name__ == "__main__":
    desc = ("Main method for running Minml pipeline")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', dest='config', type=str,
                        help=('Model config path'))
    parser.add_argument('--load_db', dest='load_db', type=bool, default=0,
                        help=('Select this option to initalize DB'))
    args = parser.parse_args()

    m = Minml(str(args.config), bool(args.load_db))
    m.run()
