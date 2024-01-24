
### Business Understanding

There are a lot of marketing campaigns that target general public. And these days to enhance the success of these campaigns, more and more marketing campaigns are being targetted directly to the consumer. Goal of this exercise is to find a model that can clearly categorize a customer into whether they will be subscribing to a bank deposit or not. This information is captured in the target variable 'y' in the dataset with values as 'yes' or 'no'.
There are a total of 41187 samples in the data with 19 features plus the target feature. 

Dataset (bank data) that was provided as a csv file was loaded as a dataframe. Following is the data model for the data. You can see that there are some categorical and numerical features in the dataset.

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 41188 entries, 0 to 41187
Data columns (total 21 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   age             41188 non-null  int64  
 1   job             41188 non-null  object 
 2   marital         41188 non-null  object 
 3   education       41188 non-null  object 
 4   default         41188 non-null  object 
 5   housing         41188 non-null  object 
 6   loan            41188 non-null  object 
 7   contact         41188 non-null  object 
 8   month           41188 non-null  object 
 9   day_of_week     41188 non-null  object 
 10  duration        41188 non-null  int64  
 11  campaign        41188 non-null  int64  
 12  pdays           41188 non-null  int64  
 13  previous        41188 non-null  int64  
 14  poutcome        41188 non-null  object 
 15  emp.var.rate    41188 non-null  float64
 16  cons.price.idx  41188 non-null  float64
 17  cons.conf.idx   41188 non-null  float64
 18  euribor3m       41188 non-null  float64
 19  nr.employed     41188 non-null  float64
 20  y               41188 non-null  object 
dtypes: float64(5), int64(5), object(11)


### Exploratory Data Analysis

- Data did not have any missing data but there were about 12 duplicate records in the data, which were dropeed resulting in total samples to 41176
- Ran different plots to analyze all features in the dataset. Observations and subsequent actions are listed in Data Preperation section.
- Initial analysis, through heat map, clearly suggested that target variable is highly correlated to 'duration' of the call, 'previous' feature was also highly correlated, which suggests that if the customer was previously connected with.
- Only 11% of the contacts had a success rate

### Data Preperation
- There are about 12 duplicate records that were dropped. Based on the directions in the Practical Assignment, ran models on only the bank information features (age, job, marital, education, default, housing and loan) to predict the target variable (y)

### Updating Categorical Columns and creating training/test data sets.

- Split the prepared data into features (X) and target (y).
- Creating training and test data sets from original data set, with a split of 70/30
- Categorical Columns and Numerical Columns to encode the data so that categorical values will be converted to numericals using category_encoders and then scaled so that they can be used for ML operations can be performed


## Modeling
- Once final dataset is ready, ran four different machine learning models, Logistic Regression, Decision Trees, K Nearest Neighbor and Support Vector Machines. Following are the results of my models:

                   Model  Train Time  Train Accuracy  Test Accuracy
0          Baseline Model    0.282880        0.887378       0.887270
1     Logistic Regression    0.163853        0.887343       0.887351
2     K Nearest Neighbour    0.059141        0.891748       0.876345
3           Decision Tree    0.099768        0.889702       0.884276
4  Support Vector Machine   64.544134        0.887343       0.887351


### Evaluation of Models
Based on the results provided, we can evaluate the models as follows:

Baseline Model: This model is the simplest model that predicts the majority class for all samples. It has a train accuracy of 0.887378 and a test accuracy of 0.887270, which are both close to the accuracy of the other models. However, this model does not provide any insights or improvements over simply guessing the majority class.

Logistic Regression: This model has a train accuracy of 0.887343 and a test accuracy of 0.887351, which are similar to the baseline model. However, logistic regression is a more sophisticated model that can provide insights into the relationship between the features and the target variable.

K Nearest Neighbour: This model has a train accuracy of 0.891748 and a test accuracy of 0.876345, which are lower than the other models. This suggests that KNN may not be the best model for this problem, as it is sensitive to noise and outliers in the data.

Decision Tree: This model has a train accuracy of 0.889702 and a test accuracy of 0.884276, which are slightly lower than the baseline and logistic regression models. However, decision trees can be useful for understanding the relationships between features and the target variable, as they can be visualized and interpreted.

Support Vector Machine: This model has a train accuracy of 0.887343 and a test accuracy of 0.887351, which are similar to the baseline and logistic regression models. However, SVMs can be computationally expensive and may not be practical for large datasets.

Overall, based on these results, we might choose logistic regression as the best model for this problem, as it provides good accuracy and insights into the relationship between features and the target variable. However, it's important to note that these results may not generalize to other datasets or problems, and it's always a good idea to try multiple models and evaluate their performance on different metrics before making a final decision.

## Final Results & Recommendations

- Duration of the call is an important feature in predicting if a customer will subscribe, this is based on the heat map. From models perspective, based on training and test accuracy, all models performed similarly (0.888 and 0.887 respectively) but Logisitc regression took significantly lesser time and SVM took the longest time.

Upon fine tuning the models with GridSearch, performance of Decision Tree classifier with a maximum depth of 3 has the best time with similar training, test accuracy. Precision score, recall score and F1 score of the models was also very similar. Overall results can be seen below.

Training SVM...
Training time: 172.448s
Best parameters: {'C': 0.1, 'kernel': 'linear'}
Training accuracy: 0.888
Test accuracy: 0.887
Precision Score: 0.4583333333333333
Recall Score: 0.023529411764705882
F1 Score: 0.04476093591047813

Training Decision Tree...
Training time: 0.595s
Best parameters: {'max_depth': 3}
Training accuracy: 0.888
Test accuracy: 0.887
Precision Score: 0.4583333333333333
Recall Score: 0.023529411764705882
F1 Score: 0.04476093591047813

Training Logistic Regression...
Training time: 2.971s
Best parameters: {'C': 0.1, 'penalty': 'l2'}
Training accuracy: 0.888
Test accuracy: 0.887
Precision Score: 0.4583333333333333
Recall Score: 0.023529411764705882
F1 Score: 0.04476093591047813

Training KNeighbor ...
Training time: 15.453s
Best parameters: {'C': 0.1, 'penalty': 'l2'}
Training accuracy: 0.888
Test accuracy: 0.887
Precision Score: 0.4583333333333333
Recall Score: 0.023529411764705882
F1 Score: 0.04476093591047813