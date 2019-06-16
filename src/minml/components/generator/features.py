import click

class FeatureGenerator():
    def __init__(self, params):
        self.params = params


    def transform(self, data):
        click.echo(f"Starting feature generation")
        print(self.params)
