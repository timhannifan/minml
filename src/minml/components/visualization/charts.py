

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import (plot_predicted_scores,precision_recall_curve, save_fig, AXIS, TITLE, TICKS)



class ChartMaker(object):
    """docstring for ChartMaker"""
    def __init__(self, config, viz_dir):
        self.config = config
        self.viz_dir = viz_dir

    def plot_pr(self, data):
        """
        Generates precision/recall graphs

        Inputs:
            - data (tuple): paramaters for visualisation. see params below
        Returns:
            Nothing
        """

        if 'generate_graphs' in self.config and self.config['generate_graphs']:
            y_true, y_score, baseline, dir_path, title = data
            self.plot_precision_recall(y_true, y_score, baseline, dir_path, title)


    def plot_precision_recall(self, y_true, y_score, baseline, dir_path, title=""):
        """
        Generates plots for precision and recall curve. This function is
        adapted from https://github.com/rayidghani/magicloops.

        Inputs:
            - y_true (Series): the Series of true target values
            - y_score (Series): the Series of scores for the model
            - baseline (float): the proportion of positive observations in the
                sample
            - dir_path (str): path of the directory for training visualization
            - title (string): the name of the model

        """
        print('saving ', title)
        pr, re, thresholds = precision_recall_curve(y_true, y_score)
        pr = pr[:-1]
        re = re[:-1]
        pct_above_per_thresh = []
        number_scored = len(y_score)

        for value in thresholds:
            num_above_thresh = len(y_score[y_score >= value])
            pct_above_thresh = num_above_thresh / float(number_scored)
            pct_above_per_thresh.append(pct_above_thresh)

        pct_above_per_thresh = np.array(pct_above_per_thresh)

        plt.clf()
        fig, ax1 = plt.subplots()
        ax1.plot(pct_above_per_thresh, pr, 'b')
        ax1.set_xlabel('Percent of Population', fontproperties=AXIS)
        ax1.set_ylabel('Precision', fontproperties=AXIS, color='b')

        ax2 = ax1.twinx()
        ax2.plot(pct_above_per_thresh, re, 'r')
        ax2.set_ylabel('Recall', fontproperties=AXIS, color='r')
        plt.title("Precision, Recall, and Percent of Population\n" + title,
                  fontproperties=AXIS)

        plt.axhline(baseline, ls='--', color='black')

        ax1.set_ylim([0, 1.05])
        ax2.set_ylim([0, 1.05])
        ax1.set_xlim([0, 1])

        save_fig(dir_path, "/precision_recall/", title, fig)


    def plot_auc_roc(self, clf, X_train, X_test, y_train, y_test, dir_path, title=""):
        """
        Plot the AUC ROC curve of the specific classifier.

        Inputs:
            - clf: a classifier
            - X_train (NumPy array): training data feature matrix
            - X_test (NumPy array): test data feature matrix
            - y_train (NumPy array): training data target matrix
            - y_test (NumPy array): test data target matrix
            - dir_path (str): path of the directory for training visualization
            - title (string): the name of the model

        """
        y_train = label_binarize(y_train, classes=[0, 1, 2])
        y_test = label_binarize(y_test, classes=[0, 1, 2])

        classifier = OneVsRestClassifier(clf)
        if hasattr(classifier, "decision_function"):
            y_score = classifier.fit(X_train, y_train).decision_function(X_test)
        else:
            y_score = classifier.fit(X_train, y_train).predict_proba(X_test)

        # Compute ROC curve and ROC area for binary classes
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(N_CLASSES):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        fig, _ = plt.subplots()
        plt.plot(fpr[POSITIVE], tpr[POSITIVE], color='darkorange', lw=1.5,
                 label='ROC curve (area = {:.4f})'.format(roc_auc[POSITIVE]))
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])

        plt.xlabel('False Positive Rate', fontproperties=AXIS)
        plt.ylabel('True Positive Rate', fontproperties=AXIS)
        plt.title('Receiver Operating Characteristic Curve\n' + title,
                  fontproperties=AXIS)
        plt.legend(loc="lower right")

        save_fig(dir_path, "/auc_roc/", title, fig)


    def plot_feature_importances(self, importances, col_names, dir_path, top_n, title=""):
        """
        Plot the feature importance of the classifier if it has this attribute. This
        credit to the University of Michigan.

        Inputs:
            - importances (array of floats): feature importances
            - col_names (list of strings): feature names
            - dir_path (str): path of the directory for training visualization
            - top_n (int): number of features with the highest importances to keep
            - title (string): the name of the model

        """
        indices = np.argsort(importances)[::-1][:top_n]
        labels = col_names[indices][::-1]

        fig, _ = plt.subplots(figsize=[12, 8])
        plt.barh(range(top_n), sorted(importances, reverse=True)[:top_n][::-1],
                 color='g', alpha=0.4, edgecolor=['black'] * top_n)

        plt.xlabel("Feature Importance", fontproperties=AXIS)
        plt.ylabel("Feature Name", fontproperties=AXIS)
        plt.yticks(np.arange(top_n), labels, fontproperties=AXIS)

        save_fig(dir_path, "/feature importance/", title, fig)

