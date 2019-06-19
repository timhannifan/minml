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

        # print('before', df.shape)
        for task in self.feature_config:
            for task_type, target_list in task.items():
                if task_type == 'categoricals':
                    print('before_cat', df.shape)
                    df = self.process_cat(target_list, df)
                    print('after_cat', df.shape)
                elif task_type == 'numeric':
                    df = self.process_num(target_list, df)
                elif task_type == 'binary':
                    df = self.process_bin(target_list, df)
        # print('after', df.shape)
        df.drop(['entity_id','city','state','county','primary_subject',
            'start_time', 'end_time','date','school_charter','eligible_double_your_impact_match','reach'], axis=1,inplace=True)
        return df

        # ct.fit_transform(df)
        # print(df['price'].mean())
        # for col in numeric_features:

        # for transf_type in self.params:
        #     for t_type, type_params in transf_type.items():
        #         print(t_type)
        #         for some_dict in type_params:

        #             col_target = some_dict['column']
        #             imp_dict = some_dict['imputation']
        #             strategy = some_dict['metrics']
        #             print('column', col_target)
        #             print('imputation dict', imp_dict)
        #             print('strategy',strategy)

        #             choices_sql = df[col_target].unique()

    def process_cat(self, target_list, df):
        print('processing categoricals')
        for col in target_list:
            col_name = col['column']

            df = pd.concat([df,
                            pd.get_dummies(df[col_name], prefix=col_name)],
                            axis=1)
            df.drop(col_name, axis=1, inplace=True)
        return df



    def scale_numeric_col(self, df, col_name):
        print('scale_numeric_col: ', col_name)
        arr = np.array(df[col_name])
        reshaped = arr.reshape((arr.shape[0], 1))
        scaler.fit(reshaped)
        df[col_name] = scaler.transform(reshaped)

        return df



    def process_num(self, target_list, df):
        print('processing numerics')
        for col in target_list:
            col_name = col['column']
            impute_dict = col['imputation']
            metrics_list = col['metrics']
            scale_after = col['scale']

            print('doing feature stuff on ', col_name)

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
