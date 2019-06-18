## get best performing model types by thresh

select threshold, metric, model_name, max(metric_value) as max_val from results
group by metric, threshold, model_name
order by threshold asc, max_val desc