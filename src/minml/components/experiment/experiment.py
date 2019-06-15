import random
import logging
import click
from components.timechop import Timechop



class Experiment():
    def __init__(self, config):
        self.config = config
        random.seed(config['random_seed'])
        self.initialize_components()

    def initialize_components(self):
        split_config = self.config["temporal_config"]
        self.chopper = Timechop(**split_config)
        result = self.chopper.chop_time()
        # print(result)
        for res in result:
            # print(res['train_matrix'].keys())
            for i in res['train_matrix'].items():
                print(i)

        if "label_config" in self.config:

            label_config = self.config["label_config"]
            label_name = label_config.get('name', None)
            click.echo(f"Generate labels on column: %s" % label_name)
            # self.label_generator = LabelGenerator(
            #     label_name=label_name,
            #     replace=True
            # )
        else:
            logging.warning(
                "label_config missing or unrecognized. Without labels, "
                "you will not be able to make matrices."
            )


    def run(self):
        pass
        # print('running single', self.config)




