from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

def get_date_splits(config):
    """
    Produces a list of date splits for an experiment using the exp config.
    Inputs:
        - config (dict): temporal_config section of config.yaml
    Returns:
        (list): Tuples of each split (train_st, train_end, test_st, test_end)
    """

    # start time of our data (yyyy-mm-dd)
    start_time = config['start_time']
    #last date of data including labels and outcomes that we have (yyyy-mm-dd)
    end_time = config['end_time']
    #how far out do we want to predict (months)
    prediction_windows = config['prediction_windows']
    #how often is this prediction being made? every day? every month? once a year? (months)
    model_update_frequency = config['model_update_frequency']
    #how long is the gap between the end of the training data and start of test, where the outcome is unknown? (days)
    unknown_outcome_gap = config['unknown_outcome_gap']
    # over what length of time should we calculate aggregations
    aggregation_lengths = config['aggregation_lengths']

    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    train_start_time = start_time_date
    res = []

    for prediction_window in prediction_windows:
        test_end_time = end_time_date
        while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
            test_start_time = test_end_time - relativedelta(months=+prediction_window)
            train_end_time = test_start_time  - relativedelta(days=+unknown_outcome_gap)

            res.append((train_start_time,train_end_time,test_start_time,test_end_time))
            test_end_time -= relativedelta(months=+model_update_frequency)
    return res
