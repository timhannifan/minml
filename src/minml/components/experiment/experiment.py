import csv
import random
import logging
import click
import importlib
import statistics

from .db_engine import DBEngine
from .time_utils import get_date_splits
from components.generator.features import FeatureGenerator
from components.visualization import plot_precision_recall

import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)

import warnings
from sklearn.exceptions import (DataConversionWarning, ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

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
        strings = ','.join([str(x) for x in row])

        with open(self.config['output_path'], "a+", newline='') as f:
            f.write(strings+'\n')


    def plot_pr(self, data):
        """
        Generates precision/recall graphs

        Inputs:
            - data (tuple): paramaters for visualisation. see params below
        Returns:
            Nothing
        """
        if self.config['generate_graphs']:
            y_true, y_score, baseline, dir_path, title = data
            plot_precision_recall(y_true, y_score, baseline, dir_path, title)


    def metrics_at_k(self, k, test_y, probs, metric_name):
        """
        Predict based on predicted probabilities and population threshold k,
        where k is the percentage of population at the highest probabilities to
        be classified as "positive". Label those with predicted probabilities
        higher than (1- k/100) quantile as positive, and evaluate the precision.
        Orginally written by https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/train.py
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

        best_at_k = 0.0
        best = []
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
                            train_info = (test_y,
                                probs,
                                0.3,
                                self.config['viz_path'],
                                "%s: %s" % (sk_model, str(params))
                                )
                            if m == 'precision':
                                if best_at_k == 0:
                                    best_at_k = m_at_k
                                    best.append((report, train_info))
                                elif m_at_k == best_at_k:
                                    best.append((report, train_info))
                                elif m_at_k > best_at_k:
                                    best_at_k = m_at_k
                                    best = [(report, train_info)]
                            # writes to results table
                            self.dbclient.write_result(report)
                            # writes to results.csv
                            self.write_result(report)


                # if 'top_n' in thresh:
                #     for n in thresh['top_n']:
                #         for m in metrics:
                #             pass

            # else:
            #     print('no thresholds', metric)
            #     pass


        return best

    def get_predicted_probabilities(self, clf, test_x):
        """
        Apply the fitted classifier on validation sets and get predicted
        probabilities for each observation. For classifiers that cannot
        provide predicted probabilities, get its standardized decision
        function.
        Orginally written by https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/train.py
        Returns:
            (array of floats) predicted probabilities
        """
        if hasattr(clf, "predict_proba"):
            predicted_prob = clf.predict_proba(test_x)[:, 1]
        else:
            prob = clf.decision_function(test_x)
            predicted_prob = (prob - prob.min()) / (prob.max() - prob.min())

        return predicted_prob

    def train(self, sk_model, params, data):
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
            split_best_prec = 0.0
            split_best_models = []

            train_x, train_y = self.dbclient.fetch_data(tr_s, tr_e)
            test_x, test_y = self.dbclient.fetch_data(te_s, te_e)

            # Running feature transformations on train/test x data
            rich_train_x = self.feature_gen.transform(train_x)
            rich_test_x = self.feature_gen.transform(test_x)

            data = (rich_train_x, train_y, rich_test_x, test_y)

            # Iterate through config models
            for sk_model, param_dict in model_config.items():
                click.echo("Starting model: %s" % (sk_model))
                param_combinations = list(ParameterGrid(param_dict))

                # For this model, iterate through parameter combinations
                for params in param_combinations:
                    clf = self.train(sk_model, params, data)

                    y_hats = clf.predict(rich_test_x)
                    probs = self.get_predicted_probabilities(clf, rich_test_x)
                    evl = self.evaluate(clf,
                                        data,
                                        y_hats,
                                        probs,
                                        split,
                                        sk_model,
                                        params)

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
                self.plot_pr(train_info)
        click.echo(f"Experiment finished")

