import csv
import random
import logging
import click

from .db_engine import DBEngine
from .time_utils import get_date_splits
from components.generator.features import FeatureGenerator
from components.model.fitter import ModelFitter

from sklearn.model_selection import ParameterGrid

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class Experiment():
    def __init__(self, experiment_config, load_db):
        self.config = experiment_config
        self.load_db = load_db
        self.dbclient = DBEngine(self.config['project_path'],
                  self.config['input_path'])
        self.feature_gen = FeatureGenerator(self.config['feature_generation'])
        self.fitter = ModelFitter()
        self.splits = get_date_splits(self.config['temporal_config'])

        random.seed(self.config.get('random_seed', 123456))
        print(self.config['input_path'])
        if load_db:
            self.dbclient.run()


    def write_result(self, row):
        with open(self.config['output_path'], 'w', newline='') as f:
            fmanager = csv.writer(f, delimiter=' ',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fmanager.writerow(row)

    def evaluate(self, trained_mode, test_x, test_y):
        score_config = self.config['scoring']

        testing_metric_list = score_config['testing_metric_groups']

        for metric_dict in testing_metric_list:
            # precision@, recall@
            metrics = metric_dict['metrics']

            if 'thresholds' in metric_dict:
                thresh = metric_dict['thresholds']

                if 'percentiles' in thresh:
                    for p in thresh['percentiles']:
                        for m in metrics:
                            # print(m,p)
                            self.get_pct_metrics(m, p)
                if 'top_n' in thresh:
                    for n in thresh['top_n']:
                        for m in metrics:
                            # print(m,n)
                            self.get_top_n_metrics(m, n)


            else:
                print('no thresholds', metric)
                pass


    def get_pct_metrics(self, metric, pct):
        print('getting pct metrics', metric, pct)
    def get_top_n_metrics(self, metric, n):
        print('getting top n metrics', metric, n)


    def run(self):
        model_config = self.config['model_config']
        splits = self.splits

        for i, split in enumerate(splits):
            click.echo("\nStarting split: %s of %s" % (i, len(splits)))
            tr_s, tr_e, te_s, te_e = split

            train_x, train_y = self.dbclient.fetch_data(tr_s, tr_e)
            test_x, test_y = self.dbclient.fetch_data(te_s, te_e)

            # Running feature transformations on train/test x data
            rich_train_x = self.feature_gen.transform(train_x)
            rich_test_x = self.feature_gen.transform(test_x)

            # Iterate through config models
            for sk_model, param_dict in model_config.items():
                # click.echo(
                    # "\nStarting model: %s on end %s with" % (sk_model, tr_e))
                param_combinations = list(ParameterGrid(param_dict))

                # For this model, iterate through parameter combinations
                for params in param_combinations:
                    # pass
                    clf = self.fitter.train(sk_model, params, rich_train_x, train_y)

                    self.evaluate(clf, rich_test_x, test_y)

        click.echo(f"Experiment finished")


