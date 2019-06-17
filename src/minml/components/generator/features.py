import click


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
numeric_features = ['price']


from numpy import array
from numpy import reshape
# numeric_steps = [
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())]

# numeric_transformer =
ct = ColumnTransformer(
    transformers=[
        # ('imputer', SimpleImputer(strategy='median'), numeric_features),
        ('scaler', StandardScaler(copy=True, with_mean=True, with_std=True), numeric_features)])
        # ('cat', categorical_transformer, categorical_features)])

scaler = StandardScaler(copy=False, with_mean=False, with_std=True)

class FeatureGenerator():
    def __init__(self, params):
        self.params = params


    def transform(self, df):
        click.echo(f"Starting feature generation")

        arr = array(df['price'])
        data = arr.reshape((arr.shape[0], 1))
        scaler.fit(data)
        df['price'] = scaler.transform(data)

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








