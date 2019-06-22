import os
import sys
import random
import logging
import click

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
        self.save_to = self.config['project_path'] + 'results/'
        self.seed = self.config.get('random_seed', 123456)
        self.res_dir = self.save_to + 'model/'
        self.viz_dir = self.save_to + 'visualization/'
        self.save_feature_data_to = self.save_to + 'feature_data/'
        delete_on_gen = [self.res_dir,self.viz_dir]
        if self.config.get('drop_existing_train_test'):
            delete_on_gen.append(self.save_feature_data_to)
        self.generate_project_dirs(delete_on_gen)

        random.seed(self.seed)
        self.dbclient = DBEngine(self.config['project_path'],
                  self.config['input_path'], self.db_config)
        self.feature_gen = FeatureGenerator(self.config['feature_generation'],
                                            self.seed)
        self.evaluator = ModelEvaluator(self.config, self.dbclient, self.seed,
                                        self.res_dir)
        self.splits = get_date_splits(self.config['temporal_config'])
        self.chartmaker = ChartMaker(self.config,self.viz_dir)
        self.tt_names = [('train','x'),('train','y'),('test','x'), ('test','y')]

        if load_db:
            self.dbclient.run()

    def generate_project_dirs(self, to_generate):
        '''
        Creates directories for results and visualizations, erases directory
        content for each run of Experiment
        Inputs:
        to_generate (list): list of directories to delete/create
        Returns: nothing
        '''
        if not os.path.exists(self.save_to):
            os.makedirs(self.save_to)

        for path in to_generate:
            if not os.path.exists(path):
                os.makedirs(path)

            for f_name in os.listdir(path):
                f_path = os.path.join(path, f_name)
                try:
                    if os.path.isfile(f_path):
                        os.unlink(f_path)
                except Exception as e:
                    print(e)

    def build_train_test(self, split):
        train_start, train_end, test_start, test_end = split

        tr_col_names, train = self.dbclient.get_split(train_start,
                                                      train_end)
        te_col_names, test = self.dbclient.get_split(test_start, test_end)

        train_df = pd.DataFrame(train, columns=tr_col_names)
        train_x = train_df.drop('result', axis=1)
        train_y = train_df['result']

        test_df = pd.DataFrame(test, columns=te_col_names)
        test_x = test_df.drop('result', axis=1)
        test_y = test_df['result']

        print('train_x, train_y shapes: ',train_x.shape, train_y.shape)
        print('test_x, test_y shapes: ',test_x.shape, test_y.shape)

        #Temporarily concat train/test to run through featuregen
        train_x['temp_label'] = 'train'
        test_x['temp_label'] = 'test'
        concat_df = pd.concat([train_x, test_x])

        features_df = self.feature_gen.transform(concat_df)
        features_df.reset_index(drop=True, inplace=True)

        # Split back to train/test, Drop temp label
        train_x = features_df[features_df['temp_label'] != 'test']
        test_x = features_df[features_df['temp_label'] == 'test']
        train_x.drop('temp_label', axis=1, inplace=True)
        test_x.drop('temp_label', axis=1, inplace=True)

        return (train_x, train_y, test_x, test_y)

    def save_train_test(self, dfs, split):

        if not os.path.exists(self.save_feature_data_to):
            os.makedirs(self.save_feature_data_to)


        def _write(i, df, split):
            # df.to_csv(self.save_feature_data_to +
                      # self.tt_names[i]+'_' + str(split) + '.csv')
            df.to_csv(self.feature_fname(i, self.tt_names[i][0],
                                             self.tt_names[i][1]),
                      header=True)
        for i, df in enumerate(dfs):
            _write(i, df, split)

    def feature_fname(self, split_num, te_or_tr, x_or_y):
        return '%s%s_%s_%s.csv'%(self.save_feature_data_to, te_or_tr,
                                 x_or_y, split_num)

    def process_best_models(self, split_best_models):
        # Generate graphs for the best models in the split
        for model in split_best_models:
            test_y = model.get('test_data').get('y')
            y_score = model.get('test_data').get('probs')
            baseline = model.get('results').get('baseline')
            dir_path = self.viz_dir
            title = model.get('meta').get('title')
            train_x = model.get('train_data').get('x')
            train_y = model.get('train_data').get('y')
            clf = model.get('model').get('clf')

            if ('generate_graphs' in self.config and
            self.config['generate_graphs']):
                self.chartmaker.plot_precision_recall(test_y, y_score,
                                                      baseline, dir_path,
                                                      title)

            if ('generate_csv' in self.config and
            self.config['generate_csv']):
                pd.DataFrame(y_score).to_csv(self.res_dir+'y_score.csv')

            self.cv_scores = self.evaluator.cross_validate(clf, train_x,
                                                                train_y)


    def run(self):
        print('RUN')
        model_config = self.config['model_config']
        splits = self.splits

        for i, split in enumerate(splits):
            # if i >0:
            #     continue
            click.echo("\nStarting split: %s of %s" % (i, len(splits)))
            train_start, train_end, test_start, test_end = split
            split_best_prec = 0.0
            split_best_models = []

            if self.config.get('use_exising_train_test'):
                train_x  = pd.read_csv(self.feature_fname(i, 'train', 'x'))
                train_y  = pd.read_csv(self.feature_fname(i, 'train', 'y'))
                test_x  = pd.read_csv(self.feature_fname(i, 'test', 'x'))
                test_y  = pd.read_csv(self.feature_fname(i, 'test', 'y'))
            else:
                train_x, train_y, test_x, test_y = self.build_train_test(split)
                self.save_train_test([train_x, train_y, test_x, test_y], i)

            data = (train_x, train_y, test_x, test_y)

            # Iterate through config models
            for sk_model, param_dict in model_config.items():
                click.echo("Starting model: %s" % (sk_model))
                param_combinations = list(ParameterGrid(param_dict))

                # For this model, iterate through parameter combinations
                for params in param_combinations:
                    clf = self.evaluator.train(sk_model, params, data)
                    y_hats = self.evaluator.predict(clf, test_x)
                    probs = self.evaluator.get_predicted_probabilities(clf, test_x)
                    baseline = self.evaluator.get_baseline(train_y)
                    evl = self.evaluator.evaluate(clf, data, y_hats, probs,
                        split, sk_model, params, baseline)

                    for best in evl:
                        if (len(set(test_y) - set(y_hats)) != 0) or len(set(y_hats)) == 1:
                            continue
                        current_best_prec = best.get('results').get('metric_value')

                        if split_best_prec == 0:
                            split_best_prec = current_best_prec
                            split_best_models.append(best)

                        elif current_best_prec == split_best_prec:
                            split_best_models.append(best)

                        elif current_best_prec > split_best_prec:
                            split_best_prec = current_best_prec
                            split_best_models = [best]

            print('Number of best splits',len(split_best_models))
            self.process_best_models(split_best_models)

