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



### Comparing Models Across Metrics
Under construction. Top performing models are presented below.
![KNN:](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/knn.png)
![Bagging:](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/bagging.png)
![SVM:](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/svm.png)
![Random Forest:](https://github.com/timhannifan/minml/blob/master/examples/donors/sample_results/sample_images/random_forest.png)

### Analyzing Results Over Time
Under construction.

### Deployment Recommendation
Under construction.

- robustness to outliers
- stability over splits
- future-proofing
