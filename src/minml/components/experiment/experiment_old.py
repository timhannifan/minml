import random
import logging
import click
from components.timechop import Timechop

import pandas as pd
import csv



class Experiment():
    def __init__(self, arg_dict):
        self.arg_dict = arg_dict

        random.seed(self.arg_dict.get('random_seed', 123456))

        self.read_data()
        self.initialize_components()

    def initialize_components(self):
        click.echo(f"Initializing components")

        self.chopper = Timechop(**self.arg_dict["temporal_config"])
        self.splits = self.chopper.chop_time()
        self.labels = self.generate_labels()

    def read_data(self):
        click.echo(f"Reading data")
        path = self.arg_dict['input_path']
        self.raw_df = pd.read_csv(path)

        #self.write_result(df.iloc[0])

    def write_result(self, row):
        with open(self.arg_dict['output_path'], 'w', newline='') as f:
            fmanager = csv.writer(f, delimiter=' ',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fmanager.writerow(['Spam'] * 5 + ['Baked Beans'])
            fmanager.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    def generate_labels(self):
        label_config = self.arg_dict.get('label_config')

        click.echo(f"Generate labels on column: %s" % label_config['name'])

    def run(self):
        pass

