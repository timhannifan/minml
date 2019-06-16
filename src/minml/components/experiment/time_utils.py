
def get_date_splits(config):
    print(config)
    # start time of our data
    start_time = config['start_time']

    #last date of data including labels and outcomes that we have
    end_time = config['end_time']

    #how far out do we want to predict (let's say in months for now)
    prediction_windows = config['prediction_windows']

    #how often is this prediction being made? every day? every month? once a year?
    model_update_frequency = config['model_update_frequency']

    from datetime import date, datetime, timedelta
    from dateutil.relativedelta import relativedelta

    start_time_date = datetime.strptime(start_time, '%Y-%m-%d')
    end_time_date = datetime.strptime(end_time, '%Y-%m-%d')
    train_start_time = start_time_date
    res = []
    for prediction_window in prediction_windows:
        test_end_time = end_time_date
        while (test_end_time >= start_time_date + 2 * relativedelta(months=+prediction_window)):
            test_start_time = test_end_time - relativedelta(months=+prediction_window)
            train_end_time = test_start_time  - relativedelta(days=+1) # minus 1 day

            res.append((train_start_time,train_end_time,test_start_time,test_end_time))
            test_end_time -= relativedelta(months=+model_update_frequency)
    return res
