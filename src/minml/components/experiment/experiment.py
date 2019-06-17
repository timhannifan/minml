import csv
import random
import logging
import click

from .db_engine import DBEngine
from .time_utils import get_date_splits
from components.generator.features import FeatureGenerator
from components.model.fitter import ModelFitter

from sklearn.model_selection import ParameterGrid




class Experiment():
    def __init__(self, experiment_config, load_db):
        self.config = experiment_config
        self.load_db = load_db
        self.dbclient = DBEngine(self.config['project_path'],
                  self.config['input_path'])
        self.feature_gen = FeatureGenerator(self.config['feature_generation'])
        self.fitter = ModelFitter()

        random.seed(self.config.get('random_seed', 123456))

        if load_db:
            self.dbclient.run()

        self.initialize_components()


    def initialize_components(self):
        click.echo(f"Initializing components")

        self.splits = get_date_splits(self.config['temporal_config'])

        if self.load_db:
            self.generate_db()


    def write_result(self, row):
        with open(self.config['output_path'], 'w', newline='') as f:
            fmanager = csv.writer(f, delimiter=' ',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fmanager.writerow(row)


    def run(self):
        model_config = self.config['model_config']
        splits = self.splits

        for i, split in enumerate(splits):
            click.echo("\nStarting split: %s of %s" % (i, len(splits)))
            tr_s, tr_e, te_s, te_e = split

            train_x, train_y = self.dbclient.fetch_data(tr_s, tr_e)
            test_x, test_y = self.dbclient.fetch_data(te_s, te_e)

            # Features generated on training only
            rich_train = self.feature_gen.transform(train_x)

            # Iterate through config models
            for sk_model, param_dict in model_config.items():
                click.echo(
                    "\nStarting model: %s on end %s with" % (sk_model, tr_e))
                param_combinations = list(ParameterGrid(param_dict))

                # For this model, iterate through parameter combinations
                for params in param_combinations:
                    self.fitter.train(sk_model, params, train_x, train_y)

        click.echo(f"Experiment finished")


