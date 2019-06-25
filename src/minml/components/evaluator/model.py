import logging
import click
import importlib

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import cross_validate

KFOLD_NUMBER = 3

class ModelEvaluator(object):
    """docstring for ModelEvaluator"""
    def __init__(self, config, dbclient, random_seed, res_dir):
        self.config = config
        self.dbclient = dbclient
        self.res_dir = res_dir
        self.metric_map = {"accuracy": accuracy_score,
                            "precision": precision_score,
                            "recall": recall_score,
                            "f1": f1_score,
                            "roc_auc": roc_auc_score}
        if random_seed is not None:
            self.seed = random_seed

    def train(self, sk_model, params, data):
        rich_train_x, train_y, rich_test_x, test_y = data
        # print('trainycols', train_y.head())
        # train_y.reset_index(drop=True, inplace=True)
        # test_y = test_y.reset_index(drop=True, inplace=True)

        # print(type(train_y))

        module_name, class_name = sk_model.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**params)

        return instance.fit(rich_train_x, train_y)

    def get_baseline(self, y_data):
        return y_data[y_data == 1].shape[0]/y_data.shape[0]

    def predict(self, clf, x_df):
        return clf.predict(x_df)

    def evaluate(self,clf, data, y_hats, probs, split, sk_model, params, baseline):
        score_config = self.config['scoring']
        test_metric_groups = score_config['testing_metric_groups']
        train_x, train_y, test_x, test_y = data
        tr_s, tr_e, te_s, te_e = split
        best_at_thresh = 0.0
        best = []

        for group in test_metric_groups:
            metrics = group['metrics']

            if 'thresholds' in group:
                if 'percentiles' in group['thresholds']:
                    for pct in group['thresholds']['percentiles']:
                        for m in metrics:
                            m_at_thresh = self.metrics_at_thresh(pct, test_y,
                                                              probs, m)
                            report = [tr_s, tr_e, te_s, te_e, sk_model,
                                      params, m, pct, m_at_thresh]

                            eval_dict = {
                                'report': report,
                                'results': {
                                    'metric': m,
                                    'threshold': pct,
                                    'metric_value': m_at_thresh,
                                    'baseline': baseline
                                },
                                'model': {
                                    'type': sk_model,
                                    'params': params,
                                    'clf': clf
                                },
                                'train_data':{
                                    'x': train_x,
                                    'y': train_y

                                },
                                'test_data': {
                                    'x': test_x,
                                    'y': test_y,
                                    'probs': probs
                                },
                                'meta': {
                                    'title': sk_model + str(params)
                                }
                            }
                            if m == 'precision':
                                if best_at_thresh == 0:
                                    best_at_thresh = m_at_thresh
                                    best.append(eval_dict)
                                elif m_at_thresh == best_at_thresh:
                                    best.append(eval_dict)
                                elif m_at_thresh > best_at_thresh:
                                    best_at_thresh = m_at_thresh
                                    best = [eval_dict]

                            self.dbclient.write_result(report)


                elif 'top_n' in group['thresholds']:
                    for n in group['thresholds']['top_n']:
                        print('Calculating top_n', n)


        print('MODEL EVALUATE FOUND %s BEST MODELS'%(len(best)))
        return best

    def cross_validate(self, clf, train_x, train_y):
        metrics = ['precision', 'recall','roc_auc']

        return cross_validate(clf, train_x, train_y, scoring=metrics,
                                cv=KFOLD_NUMBER, return_train_score=True)



    def metrics_at_thresh(self, k, test_y, probs, metric_name):
        """
        Predict based on predicted probabilities and population threshold k,
        where k is the percentage of population at the highest probabilities to
        be classified as "positive". Label those with predicted probabilities
        higher than (1- k/100) quantile as positive, and evaluate the precision.
        Orginally written by https://github.com/KunyuHe/ML-Pipeline-for-Crowdfunding-Project-Outcome-Prediction/blob/master/codes/train.py
        Inputs:
            - test_y (array): of true labels.
            - probs (array): predicted probabilities on the validation
                set.
            metic_name: (str) 'precision', 'recall', 'f1', 'roc_auc'
        Returns:
            (float) metric score of model at population threshold k.
        """
        idx = np.argsort(probs)[::-1]
        sorted_prob, sorted_test_y = probs[idx], test_y[idx]

        cutoff_index = int(len(sorted_prob) * (k / 100.0))
        predictions_at_thresh = [1 if x < cutoff_index else 0 for x in
                            range(len(sorted_prob))]

        skmetric = self.metric_map[metric_name]
        if metric_name == "roc_auc":
            return skmetric(sorted_test_y, sorted_prob)
        else:
            return skmetric(sorted_test_y, predictions_at_thresh)


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

