import click

class FeatureGenerator():
    def __init__(self, params):
        self.params = params


    def transform(self, df):
        click.echo(f"Starting feature generation")

        for transf_type in self.params:
            for k,v in transf_type.items():
                print(k)
                for i in v:
                    print(i)
        # print(df.describe())

        # print(df.columns)
        # return df
