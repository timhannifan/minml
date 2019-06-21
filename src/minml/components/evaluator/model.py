import logging
import click
import importlib

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)


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
                                baseline,
                                self.res_dir,
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

                            self.dbclient.write_result(report)


                # if 'top_n' in thresh:
                #     for n in thresh['top_n']:
                #         for m in metrics:
                #             pass

            # else:
            #     print('no thresholds', metric)
            #     pass


        return best

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

