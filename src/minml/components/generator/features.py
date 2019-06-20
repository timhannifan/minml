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
    def __init__(self, feature_config):
        self.feature_config = feature_config


    def transform(self, df):
        click.echo(f"Starting feature generation")
        print('start df.shape', df.shape)

        for task in self.feature_config:
            for task_type, target_list in task.items():

                if task_type == 'categoricals':
                    df = self.process_cat(target_list, df)
                elif task_type == 'numeric':
                    df = self.process_num(target_list, df)
                elif task_type == 'binary':
                    df = self.process_binary(target_list, df)
                elif task_type == 'drop':
                    df.drop(target_list, axis=1,inplace=True)

        print('end df.shape', df.shape)
        return df

    def process_cat(self, target_list, df):
        print('processing CATEGORICAL values')

        for col in target_list:
            col_name = col['column']
            df = self.impute_na(df, col_name,
                                col['imputation'], 'categorical')
            df = pd.concat([df, pd.get_dummies(df[col_name],
                            prefix=col_name)], axis=1)
            df.drop(col_name, axis=1, inplace=True)
        return df

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
        print('scale_numeric_col: ', df[col_name].shape)


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
                df[df[col_name].isna()] = config['fill_value']

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

        else:
            print('No missing values', col_name)
        return df



