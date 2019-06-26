select train_start, train_end, model_name, params as parameters, threshold as pct_threshold, round(metric_value,4) as precision
from results
where metric = 'precision' and threshold = 5
order by metric_value desc;
