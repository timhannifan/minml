import csv
import random
import logging
import click
import importlib
import statistics

from .db_engine import DBEngine
from .time_utils import get_date_splits
from components.generator.features import FeatureGenerator
from components.evaluator.model import ModelEvaluator
from components.visualization.charts import ChartMaker

import pandas as pd
from sklearn.model_selection import ParameterGrid

import warnings
from sklearn.exceptions import (DataConversionWarning, ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

class Experiment():
    def __init__(self, experiment_config, db_config, load_db):
        self.config = experiment_config
        self.db_config = db_config
        self.load_db = load_db
        self.seed = self.config.get('random_seed', 123456)
        random.seed(self.seed)
        self.dbclient = DBEngine(self.config['project_path'],
                  self.config['input_path'], self.db_config)
        self.feature_gen = FeatureGenerator(self.config['feature_generation'],
                                            self.seed)
        self.evaluator = ModelEvaluator(self.config, self.dbclient, self.seed)
        self.splits = get_date_splits(self.config['temporal_config'])
        self.chartmaker = ChartMaker(self.config)

        if load_db:
            self.dbclient.run()

    def run(self):
        print('RUN')
        model_config = self.config['model_config']
        splits = self.splits

        for i, split in enumerate(splits):
            click.echo("\nStarting split: %s of %s" % (i, len(splits)))
            tr_s, tr_e, te_s, te_e = split
            split_best_prec = 0.0
            split_best_models = []

            train_x, train_y = self.dbclient.fetch_data(tr_s, tr_e)
            test_x, test_y = self.dbclient.fetch_data(te_s, te_e)

            #Temporarily concat train/test to run through featuregen
            train_x['temp_label'] = 'train'
            test_x['temp_label'] = 'test'
            concat_df = pd.concat([train_x, test_x])

            features_df = self.feature_gen.transform(concat_df)

            # print('train_x_df', train_x.shape)
            # print('features_df', features_df.shape)
            # print('concat_df', concat_df.shape)

            # Split back to train/test
            rich_train_x = features_df[features_df['temp_label'] == 'train']
            rich_test_x = features_df[features_df['temp_label'] == 'test']

            # Drop temp label
            rich_train_x = rich_train_x.drop('temp_label', axis=1)
            rich_test_x = rich_test_x.drop('temp_label', axis=1)

            data = (rich_train_x, train_y, rich_test_x, test_y)

            # Iterate through config models
            for sk_model, param_dict in model_config.items():
                click.echo("Starting model: %s" % (sk_model))
                param_combinations = list(ParameterGrid(param_dict))

                # For this model, iterate through parameter combinations
                for params in param_combinations:
                    clf = self.evaluator.train(sk_model, params, data)
                    y_hats = self.evaluator.predict(clf, rich_test_x)
                    probs = self.evaluator.get_predicted_probabilities(clf, rich_test_x)
                    baseline = self.evaluator.get_baseline(train_y)
                    evl = self.evaluator.evaluate(clf, data, y_hats, probs,
                        split, sk_model, params, baseline)

                    curr_prec = evl[0][0][8]

                    if split_best_prec == 0:
                        split_best_prec = curr_prec
                        for m in evl:
                            split_best_models.append(m)
                    elif curr_prec == split_best_prec:
                        for m in evl:
                            split_best_models.append(m)
                    elif curr_prec > split_best_prec:
                        split_best_prec = curr_prec
                        split_best_models = evl

            # Generate graphs for the best models in the split
            for model in split_best_models:
                report, train_info = model

                self.chartmaker.plot_pr(train_info)
        click.echo(f"Experiment finished")

