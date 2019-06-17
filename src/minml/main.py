import argparse
import yaml
import click

from components.experiment import Experiment

class Minml():
    def __init__(self, config, db_config, load_db):
        self.config = config
        self.db_config = db_config
        self.load_db = load_db

    def run(self):
        click.echo(f"Running Minml with config: {self.config}")
        with open(self.config) as f:
            experiment_con = yaml.load(f)
        with open(self.db_config) as f:
            db_con = yaml.load(f)

        exp = Experiment(
            experiment_config=experiment_con,
            db_config=db_con,
            load_db=self.load_db)
        exp.run()

if __name__ == "__main__":
    desc = ("Main method for running Minml pipeline")
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--config', dest='config', type=str,
                        help=('Model config path'))
    parser.add_argument('--db', dest='db', type=str,
                        help=('DB config path'))
    parser.add_argument('--load_db', dest='load_db', type=bool, default=0,
                        help=('Select this option to initalize DB'))
    args = parser.parse_args()

    m = Minml(str(args.config), str(args.db), bool(args.load_db))
    m.run()
