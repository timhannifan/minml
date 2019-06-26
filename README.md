# minml
Barebones machine learning pipeline


## Usage
Before running locally, start a Postgres database locally or remotely and add the host/domain/user/password information to `config_db.yaml`. Within the same file you can change the input_path to point to `data_full.csv` insteal of `data_small.csv`. The smaller file contains 5,000 rows and is much faster for testing in a local environment.

```
git clone https://github.com/timhannifan/minml
cd minml
sh run.sh
```

To run in ipython:
```
virtualenv env
source env/bin/activate
pip install -r requirements.txt
cd src/minml

ipython3
run main --config "../../examples/donors/config/config_test_env.yaml" --db "../../examples/donors/config/config_db.yaml"
```

In the main config file there is a `load_db` parameter that prevents table generation and re-reading csv data. After the first run, run without this set to False to improve performance. In addition, featurized datasets are saved to  the project directory using a toggle in the main config, which allows you to avoid re-generating features on a dataset.

### Requirements
See `requirements.txt` for required packages. The current stack requires:
* Python 3.6.0
* Postgres 11.3

### Configuration
Minml Experiments are configured via a set of YAML and SQL files within a project directory. See examples for test/production environments ![here](https://github.com/timhannifan/minml/tree/master/examples/donors).

The project directory should contain all the files listed in the example directory, with the same filenames. The main configuration file is `config.yaml`, which defines the Experiment parameters for creating time splits, generating features, specifying models and parameter ranges, and selecting a scoring matrix for model evaluation.

To run a custom experiment with new data, one would need to replace the code within each file prefixed with `config_` in the project directory. The SQL files define the Experiment schemas, and field-specific information for indexing, cleaning, inserting, and defining semantic entities/events.

### Experiment Results
#### Postgres Table
Results from model/parameter/threshold/time-split analysis are stored in a Postgres table named 'results'. A sample of the `best_model.sql` output is shown below.

![Postgres](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/results.png)

#### Visualizations
Precision/recall graphs vs population thresholds for the best performing models in each split are exported to a directory specified in the config. Visualizations can also be disabled in the config.

![Example:](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/knn.png)


Work is underway for further post-modeling evaluation and parameter tuning.
