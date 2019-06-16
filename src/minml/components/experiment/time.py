
def get_date_splits():
    # start time of our data
    start_time = '2012-01-01'

    #last date of data including labels and outcomes that we have
    end_time = '2014-01-01'

    #how far out do we want to predict (let's say in months for now)
    prediction_windows = [6]

    #how often is this prediction being made? every day? every month? once a year?
    update_window = 6

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
            test_end_time -= relativedelta(months=+update_window)
    return res
