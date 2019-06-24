import click
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd

scaler = StandardScaler(copy=False, with_mean=False, with_std=True)

class FeatureGenerator():
    def __init__(self, feature_config, random_seed=None):
        self.feature_config = feature_config
        if random_seed is not None:
            self.seed = random_seed
        self.one_hot_dict = {}


    def featurize_data(self, df, train_or_test):
        for task in self.feature_config:
            for task_type, target_list in task.items():
                if task_type == 'categoricals':
                    for col in target_list:
                        col_name = col['column']
                        df = self.impute_na(df, col_name,
                                            col['imputation'],
                                            'categorical')

                        df = self.process_one_hot(df, col_name,
                                                  train_or_test)
                elif task_type == 'numeric':
                    df = self.process_num(target_list, df)
                elif task_type == 'binary':
                    df = self.process_binary(target_list, df)
                elif task_type == 'drop':
                    df.drop(target_list, axis=1, inplace=True)

        return df


    def process_one_hot(self, df, col_name, train_or_test):
        reshaped = self.reshape_series(df[col_name])

        if train_or_test == 'train':
            encoder = OneHotEncoder(handle_unknown='ignore').fit(reshaped)
            raw_names = encoder.categories_
            col_names = ['%s_%s'%(col_name, x) for x in
                           raw_names[0]]
            e_props = {
                'encoder': encoder,
                'col_names': col_names
            }
            self.one_hot_dict[col_name] = e_props
        else:
            col_encoder_dict = self.one_hot_dict[col_name]
            encoder = col_encoder_dict['encoder']
            col_names = col_encoder_dict['col_names']

        labels = encoder.transform(reshaped)
        new = df.join(pd.DataFrame(labels.todense(), columns=col_names))
        new.drop(col_name, axis=1,inplace=True)

        return new


    def featurize(self, data_dict):
        trn_cols, trn_data, test_cols, test_data = tuple(data_dict.values())

        train_df = pd.DataFrame(trn_data, columns=trn_cols)
        train_y = train_df['result']
        train_x = train_df.drop('result', axis=1)
        featurized_trn_X = self.featurize_data(train_x, 'train')

        test_df = pd.DataFrame(test_data, columns=test_cols)
        test_y = test_df['result']
        test_x = test_df.drop('result', axis=1)
        featurized_test_X = self.featurize_data(test_x, 'test')

        return (featurized_trn_X, train_y, featurized_test_X, test_y)

    def process_binary(self, target_list, df):
        for col in target_list:
            col_name = col['column']
            df = self.impute_na(df, col_name, col['imputation'], 'binary')

        return df

    def process_num(self, target_list, df):
        for col in target_list:
            col_name = col['column']
            impute_dict = col['imputation']
            scale_after = col['scale']

            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            df = self.impute_na(df, col_name, impute_dict, 'numeric')

            if scale_after == True:
                self.scale_numeric_col(df, col_name)

        return df

    def scale_numeric_col(self, df, col_name):
        reshaped = self.reshape_series(df[col_name])
        scaler.fit(reshaped)
        df[col_name] = scaler.transform(reshaped)

        return df

    def reshape_series(self, series):
        arr = np.array(series)
        return arr.reshape((arr.shape[0], 1))

    def impute_na(self, df, col_name, config, f_type):
        series = df[col_name]
        missing = df[series.isna()].shape[0]

        if missing > 0:
            if f_type == 'categorical':
                df[col_name] = df[col_name].fillna(config['fill_value'])
            elif f_type == 'numeric' or f_type == 'binary':
                val_flag = np.nan
                if 'missing_values' in config:
                    val_flag = config['missing_values']

                if config['strategy'] == 'constant' and 'fill_value' in config:
                    imp_mean = SimpleImputer(missing_values=val_flag,
                                         strategy=config['strategy'],
                                         fill_value=config['fill_value'])
                else:
                    imp_mean = SimpleImputer(missing_values=val_flag,
                                         strategy=config['strategy'])

                reshaped = self.reshape_series(series)
                imp_mean.fit(reshaped)
                df[col_name] = imp_mean.transform(reshaped)

        return df

