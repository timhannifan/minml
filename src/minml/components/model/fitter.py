import click
import importlib

import pandas as pd

class ModelFitter():
    def __init__(self):
        pass

    def flattened_grid_config(self, config):
        return flatten_grid_config(config)

    def train(self, sk_model, params, train_x, train_y):
        click.echo("Starting model fit on %s" % (sk_model))

        module_name, class_name = sk_model.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**params)

        # print(type(train_x))
        df = pd.DataFrame(train_x)

        return instance.fit(train_x, train_y)

