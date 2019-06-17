import csv
import random
import logging
import click
import importlib
import statistics

from .db_engine import DBEngine
from .time_utils import get_date_splits
from components.generator.features import FeatureGenerator

import numpy as np
from sklearn.model_selection import ParameterGrid

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

import pandas as pd

class Experiment():
    def __init__(self, experiment_config, db_config, load_db):
        self.config = experiment_config
        self.db_config = db_config
        self.load_db = load_db
        self.dbclient = DBEngine(self.config['project_path'],
                  self.config['input_path'], self.db_config)
        self.feature_gen = FeatureGenerator(self.config['feature_generation'])
        self.splits = get_date_splits(self.config['temporal_config'])

        random.seed(self.config.get('random_seed', 123456))

        self.metric_map = {"accuracy": accuracy_score,
                            "precision": precision_score,
                            "recall": recall_score,
                            "f1": f1_score,
                            "roc_auc": roc_auc_score}
        if load_db:
            self.dbclient.run()


    def write_result(self, row):
        with open(self.config['output_path'], "a+", newline='') as f:
            # fmanager = csv.writer(f, delimiter=' ',
            #                 quotechar='|', quoting=csv.QUOTE_MINIMAL)
            f.write(','.join([str(x) for x in row])+'\n'
                )


    def metrics_at_k(self, k, test_y, probs, metric_name):
        """
        Predict based on predicted probabilities and population threshold k,
        where k is the percentage of population at the highest probabilities to
        be classified as "positive". Label those with predicted probabilities
        higher than (1- k/100) quantile as positive, and evaluate the precision.
        Inputs:
            - probs (array): predicted probabilities on the validation
                set.
            - test_y (array): of true labels.
        Returns:
            (float) precision score of our model at population threshold k.
        """
        idx = np.argsort(probs)[::-1]
        sorted_prob, sorted_test_y = probs[idx], test_y[idx]

        cutoff_index = int(len(sorted_prob) * (k / 100.0))
        predictions_at_k = [1 if x < cutoff_index else 0 for x in
                            range(len(sorted_prob))]

        skmetric = self.metric_map[metric_name]
        if metric_name == "roc_auc":
            return skmetric(sorted_test_y, sorted_prob)
        else:
            return skmetric(sorted_test_y, predictions_at_k)


    def evaluate(self,clf, data, y_hats, probs, split, sk_model, params):
        score_config = self.config['scoring']
        train_x, train_y, test_x, test_y = data
        tr_s, tr_e, te_s, te_e = split
        testing_metric_list = score_config['testing_metric_groups']

        for metric_dict in testing_metric_list:
            # E.g. precision@, recall@
            metrics = metric_dict['metrics']

            if 'thresholds' in metric_dict:
                thresh = metric_dict['thresholds']


                if 'percentiles' in thresh:
                    for k in thresh['percentiles']:
                        for m in metrics:
                            m_at_k = self.metrics_at_k(k, test_y, probs, m)

                            report = [tr_s, tr_e, te_s, te_e, sk_model,
                                      params, m, k, m_at_k]

                            self.write_result(report)
                # if 'top_n' in thresh:
                #     for n in thresh['top_n']:
                #         for m in metrics:
                #             pass

            else:
                print('no thresholds', metric)
                pass


    def get_predicted_probabilities(self, clf, test_x):
        """
        Apply the fitted classifier on validation sets and get predicted
        probabilities for each observation. For classifiers that cannot
        provide predicted probabilities, get its standardized decision
        function.
        Returns:
            (array of floats) predicted probabilities
        """
        print('running get_predicted_probabilities')
        if hasattr(clf, "predict_proba"):
            predicted_prob = clf.predict_proba(test_x)[:, 1]
        else:
            prob = clf.decision_function(test_x)
            predicted_prob = (prob - prob.min()) / (prob.max() - prob.min())

        return predicted_prob

    def train(self, sk_model, params, data):
        click.echo("Starting model fit on %s" % (sk_model))

        rich_train_x, train_y, rich_test_x, test_y = data

        module_name, class_name = sk_model.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**params)

        return instance.fit(rich_train_x, train_y)


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

            data = (rich_train_x, train_y, rich_test_x, test_y)
            # Iterate through config models
            for sk_model, param_dict in model_config.items():
                param_combinations = list(ParameterGrid(param_dict))

                # For this model, iterate through parameter combinations
                for params in param_combinations:
                    clf = self.train(sk_model, params, data)

                    y_hats = clf.predict(rich_test_x)
                    probs = self.get_predicted_probabilities(clf, rich_test_x)
                    self.evaluate(clf,
                                data,
                                y_hats,
                                probs,
                                split,
                                sk_model,
                                params)

        click.echo(f"Experiment finished")

