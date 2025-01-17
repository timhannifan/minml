## Donors Choose Example

### The Problem
DonorsChoose.org connects teachers in high-need communities with donors who want to help. The data used in this project covers a period from 1-Jan-2012 to 31-Dec-2013, and includes information on every project posted on Donors Choose along with demographic and geographic information about the school/teacher that created the project. The objective is to identify projects that are unlikely to be funded within the first 60 days after posting, based on the information available at prediction time.


### Time Splits
Training and testing periods are generated for every six month period spanning 1-Jan-2012 through 31-Dec-2013. This results in three train/test splits. Testing periods are separated from training by a 60-day gap to avoid including unobserved outcomes in the training data.

### Feature Engineering
Features are generated through a multistep process. The data pipeline contains these steps for each split:
1. Bulk copy csv into a Postgres table
2. Clean, convert to proper type, and store in the 'semantic' schema as either an entity or an event. Exact SQL operations are defined in config files.
3. Create training set: Loop through all feature generation tasks, as defined in the experiment configuration, and record the transformations (e.g. categorical column names) to be passed on to the test-set generation step. Feature generation tasks include imputation, scaling, and one-hot encoding. Only data known at the time prediction is contained in the training set.
4. Create the testing set: Using the transformations from the previous stage, fit the testing set to the same number of columns to encode all X variables. This approach prevents future information from being included in the training set.
5. Write the training/testing sets for the current split to disk. This is currently written to csv, but will be transitioned to Postgres in the future.
6. Repeat for all splits.


### Model Selection
The models run for this experiment were decision trees, support vector machines, logistic regression, KNN, random forests, bagging, and gradient boosting. Each model was run on a random sampling of 25% of the data across all time splits to roughly determine precision and training time.

Decision trees largely followed baseline trends across all thresholds. The baseline we used was simply the proportion of projects that didn't get funded within 60 days to the whole population of projects, which was around 30%.

Each model has it pros and cons beyond the precision/recall levels: KNN, bagging, and boosting training times are an order of magnitude longer than SVM or logistic regression. If this will be used in a development environment with strict memory\ or time constraints, it may make sense to use one of the models with shorter training times.

Early results indicated that the best candidates for further analysis would be gradient boosting, SVM and logisitic regression. These classes of models consistently performed best across all time periods, which is a good indicator of future robustness.

### Performance Analysis
Full model results are available in the Postgres table 'results'. For this experiment, we considered precision, recall, AUC, and accuracy at 1, 2, 5, 10, 20, 30, and 50 percent thresholds. The screenshot below shows our primary metric of interest, precision at 5%, from the results table. A sample output results can be found [here](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/results_sample.csv)

![](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/top_10.png)

### Precision/Recall Examples for Models

![](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/boosting.png)
![](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/svm.png)

### Parameter Grids and Fine Tuning
The experiment configuration defines which metrics and thresholds should be calculated for each model. In the case of our top performing models, the following configuration was used:
```
model_config:
    'sklearn.svm.LinearSVC':
        penalty: [l2]
        C: [1,5,10]
    'sklearn.linear_model.LogisticRegression':
        penalty: [l2]
        C: [1,5,10]
        solver: [sag]
        n_jobs: [-1]
    'sklearn.ensemble.GradientBoostingClassifier':
        n_estimators: [100,200]
        min_samples_split: [2,5]
        max_depth: [3,6]
```

In the first iteration, we define a range of parameter values to test. After selecting several models to test further, the models are refined as we set the parameters closer to their 'optimal' level. The approach taken here is somewhat crude, but it approximates what more advanced methods would do programatically.

*Visualization ond 'optimal' params coming soon.*

### Analyzing Results Over Time
The best performance, as measured by precision at the 5% threshold, was acheived on models trained on the longest period. Models trained on the shorter splits showed higher variance in their performance metrics, which makes it harder to trust one individaul reading. For robustness of prediction and reduced variance of metrics in the future, it is recommended that the training set be updated as it appears that there are increasing returns from lengthening the training set.
*Visualization coming soon*

### Deployment Recommendation
- robustness to outliers
- stability over time splits
- stability in the future
- speed and efficiency
- production environment considerations


