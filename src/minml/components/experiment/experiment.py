import csv
import random
import logging
import click

from .db_engine import DBEngine
from .time_utils import get_date_splits
from components.generator.features import FeatureGenerator
from components.model.fitter import ModelFitter




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
        for k, v in model_config.items():
            # print(type(v))

            print (k)

            for param_k, param_v in v.items():
                print(param_k, param_v)

        # splits = self.splits

        # for split in splits:
        #     click.echo("\nStarting split")
        #     tr_s, tr_e, te_s, te_e = split
        #     click.echo("Train dates: %s to %s" % (tr_s,tr_e))
        #     click.echo("Test dates: %s to %s" % (te_s,te_e))

        #     train = self.dbclient.fetch_data(tr_s, tr_e)
        #     test = self.dbclient.fetch_data(tr_s, tr_e)

        #     rich_train = self.feature_gen.transform(train)
        #     rich_test = self.feature_gen.transform(test)

        #     model_config = self.config['model_config']



        #     self.fitter.fit(train, test)



