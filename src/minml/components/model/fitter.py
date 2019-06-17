import click
import importlib

def flatten_grid_config(grid_config):
    """Flattens a model/parameter grid configuration into individually
    trainable model/parameter pairs

    Yields: (tuple) classpath and parameters
    """
    print(grid_config)
    # for class_path, parameter_config in grid_config.items():
        # print('clspath', class_path)
        # for parameters in ParameterGrid(parameter_config):
        #     yield class_path, parameters


class ModelFitter():
    def __init__(self):
        pass

    def flattened_grid_config(self, config):
        return flatten_grid_config(config)

    def fit(self, sk_model, params, train, test):
        click.echo("Starting model fit on %s" % (sk_model))

        module_name, class_name = sk_model.rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        instance = cls(**params)

        return instance.fit(train, [0]*len(train))

