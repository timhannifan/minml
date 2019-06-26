## Donors Choose Example

### The Problem
DonorsChoose.org connects teachers in high-need communities with donors who want to help. The data used in this project covers a period from 1-Jan-2012 to 31-Dec-2013, and includes information on every project posted on Donors Choose along with demographic and geographic information about the school/teacher that created the project. The objective is to identify projects that are unlikely to be funded within the first 60 days after posting, based on the information available at prediction time.

### Feature Engineering
Features are generated through a multistep process. The data pipeline contains these steps:
1. Bulk copy csv into a Postgres table
2. Clean, convert to proper type, and store in the 'semantic' schema as either an entity or an event. Exact SQL operations are defined in config files.
3. Create training set: Loop through all feature generation tasks, as defined in the experiment configuration, and record the transformations (e.g. categorical column names) to be passed on to the test-set generation step. Feature generation tasks include imputation, scaling, and one-hot encoding. Only data known at the time prediction is contained in the training set.
4. Create the testing set: Using the transformations from the previous stage, fit the testing set to the same number of columns to encode all X variables. This approach prevents future information from being included in the training set.
5. Repeat for all splits.

### Comparing Models Across Metrics

### Results Over Time

### Deployment Recommendation
