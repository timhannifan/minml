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
run main --config "../../examples/donors/config.yaml" --db "../../examples/donors/config_db.yaml" --load_db 1
```

Excluding the argument --load_db prevents table generation and re-reading the csv data. After the first run, run without --load_db to improve performance. This parameter needs to be added if it is the first run, your raw data changes, or you change any database-related configurations via SQL or YAML files.

### Requirements
See `requirements.txt` for required packages. The current stack requires:
* Python 3.6.0
* Postgres 11.3

### Configuration
Minml Experiments are configured via a set of YAML and SQL files within a project directory. See an example configuration ![here](https://github.com/timhannifan/minml/tree/master/examples/donors).

The project directory should contain all the files listed in the example directory, with the same filenames. The main configuration file is `config.yaml`, which defines the Experiment parameters for creating time splits, generating features, specifying models and parameter ranges, and selecting a scoring matrix for model evaluation.

To run a custom experiment with new data, one would need to replace the code within each file prefixed with `config_` in the project directory. The SQL files define the Experiment schemas, and field-specific information for indexing, cleaning, inserting, and defining semantic entities/events.

### Experiment Results
#### Postgres Table
Results from model/parameter/threshold/time-split analysis are stored in a Postgres table named 'results'. A sample of the `best_model.sql` output is shown below.

![Postgres](https://github.com/timhannifan/minml/blob/master/examples/donors/img/results.png)

#### Visualizations
Precision/recall graphs vs population thresholds for the best performing models in each split are exported to a directory specified in the config. Visualizations can also be disabled in the config.

Example:
![Postgres](https://github.com/timhannifan/minml/blob/master/examples/donors/visualization/precision_recall/sklearn.linear_model.LogisticRegression:%20%7B'C':%200.01%2C%20'n_jobs':%20-1%2C%20'penalty':%20'l2'%2C%20'solver':%20'sag'%7D.png)



Work is underway for further post-modeling evaluation and parameter tuning.
