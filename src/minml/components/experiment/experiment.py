import csv
import random
import logging
import click

from .experiment_db import Client
from .time import get_date_splits




class Experiment():
    def __init__(self, arg_dict, load_db):
        self.arg_dict = arg_dict
        self.load_db = load_db
        self.dbclient = Client(self.arg_dict['project_path'],
                  self.arg_dict['input_path'])

        random.seed(self.arg_dict.get('random_seed', 123456))

        if load_db:
            self.dbclient.run()

        self.initialize_components()


    def initialize_components(self):
        click.echo(f"Initializing components")

        self.splits = get_date_splits(self.arg_dict['temporal_config'])

        if self.load_db:
            self.generate_db()

        click.echo(f"Completed initializing components")



    def write_result(self, row):
        with open(self.arg_dict['output_path'], 'w', newline='') as f:
            fmanager = csv.writer(f, delimiter=' ',
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            fmanager.writerow(row)


    def run(self):
        splits = self.splits

        for split in splits:
            tr_s, tr_e, te_s, te_e = split

            train = self.dbclient.fetch_data(tr_s, tr_e)
            test = self.dbclient.fetch_data(tr_s, tr_e)

            print(len(train), len(test))


