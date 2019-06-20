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

        for task in self.feature_config:
            for task_type, target_list in task.items():
                if task_type == 'categoricals':
                    df = self.process_cat(target_list, df)
                elif task_type == 'numeric':
                    df = self.process_num(target_list, df)
                elif task_type == 'binary':
                    df = self.process_bin(target_list, df)
                elif task_type == 'drop':
                    df.drop(target_list, axis=1,inplace=True)

        return df

    def process_cat(self, target_list, df):
        print('processing categoricals')



        for col in target_list:
            col_name = col['column']

            df = pd.concat([df, pd.get_dummies(df[col_name],
                            prefix=col_name)], axis=1)
            df.drop(col_name, axis=1, inplace=True)
        return df



    def scale_numeric_col(self, df, col_name):
        print('scale_numeric_col: ', col_name)

        reshaped = self.reshape_series(df[col_name])
        scaler.fit(reshaped)
        df[col_name] = scaler.transform(reshaped)

        return df

    def reshape_series(self, series):
        arr = np.array(series)
        return arr.reshape((arr.shape[0], 1))

    def impute_na(self, df, col_name, config, num_or_cat):
        print('Processing MISSING values', col_name)

        # Check for missing values, impute if so
        missing = df[df[col_name].isna()].shape[0]

        if missing > 0:
            if num_or_cat == 'numeric':
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

                reshaped = self.reshape_series(df[col_name])
                imp_mean.fit(reshaped)
                df[col_name] = imp_mean.transform(reshaped)
        return df

    def process_num(self, target_list, df):

        for col in target_list:
            col_name = col['column']
            print('processing numeric', col_name)
            impute_dict = col['imputation']
            scale_after = col['scale']

            df = self.impute_na(df, col_name, impute_dict, 'numeric')


            # Check for missing values, impute if so
            if scale_after == True:
                self.scale_numeric_col(df, col_name)

        return df




# numeric_steps = [
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())]

# numeric_transformer =

# ct = ColumnTransformer(
#     transformers=[
#         # ('imputer', SimpleImputer(strategy='median'), numeric_features),
#         ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True), numeric_features)])
#         # ('cat', categorical_transformer, categorical_features)])
