# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 21:21:25 2019

@author: Mainak Sarkar
"""

#Churn Modelling

#Setting the working directory
import os
os.chdir("C:/Users/Mainak Sarkar/Desktop/Kaggle/Bank_Customer_Churn_Detection")

#Importing the data skipping the blank lines if any
import pandas as pd
data = pd.read_csv("Churn_Modelling.csv", skip_blank_lines = True)

#Checking the summary of the whole data set
data.describe(include = "all")

#Checking class of each column
data.dtypes





#***********************************DATA PREPROCESSING***********************************

#Data type conversion
#converting "Tenure" as factor
data['Tenure'] = data['Tenure'].astype('category')

#converting "NumOfProducts" as factor
data['NumOfProducts'] = data['NumOfProducts'].astype('category')

#converting "HasCrCard" as factor
data['HasCrCard'] = data['HasCrCard'].astype('category')

#converting "IsActiveMember" as factor
data['IsActiveMember'] = data['IsActiveMember'].astype('category')

#converting "Exited" as factor
data['Exited'] = data['Exited'].astype('category')





#***********************************MISSING VALUE ANALYSIS***********************************

#Checking null values
missing_value = pd.DataFrame(data.isnull().sum())

#Resetting Index
missing_value = missing_value.reset_index()

#Renaming Variable
missing_value = missing_value.rename(columns = {'index':'Variable Name', 0 : 'Missing-Percentage'})

#Calculating Missing Value Percentage
missing_value['Missing-Percentage'] = (missing_value['Missing-Percentage']/len(data))*100

#So the dataset has no missing values





#***********************************OUTLIER ANALYSIS***********************************

#Selecting numerical variables
numerical_cnames = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']                       #"instant" and all dependent variables are removed
#passenger_count and fare_amount is dealt separately

import numpy as np

#Detecting Outliers and replacing them with NA's
for a in numerical_cnames:
    q75, q25 = np.percentile(data.loc[:,a], [75, 25])
    iqr = q75 - q25
    min = q25 - (iqr*1.5)
    max = q75 + (iqr*1.5)
    data.loc[data[a]<min, a] = np.nan
    data.loc[data[a]>max, a] = np.nan


#Checking null values
outliers = pd.DataFrame(data.isnull().sum())

#Resetting Index
outliers = outliers.reset_index()

#Renaming Variables
outliers = outliers.rename(columns = {'index':'Variable Name', 0 : 'Missing-Percentage'})

#Calculating Missing Value Percentage
outliers['Missing-Percentage'] = (outliers['Missing-Percentage']/len(data))*100

#As we know that presence of ouliers can affect our models a lot
#So we either have to delete them or go for imputation
#But as the no. of outliers is high so we will opt for imputation





#***********************************ONLY FOR FINDING THE BEST METHOD***********************************
#Making a sample to check which method works best
#Choosing a sample and saving its value
sample_NA = data.loc[50, ['CreditScore', 'Age']]

#Putting values of sample equal to NA for required columns
data.loc[50, ['CreditScore', 'Age']] = np.nan


#MEAN Method
for b in numerical_cnames :
    data[b] = data[b].fillna(data[b].mean())
    
sample_NA_mean = data.loc[50, ['CreditScore', 'Age']]


#Re_Run the above part of code without the MEAN Method

#MEDIAN Method
for c in numerical_cnames :
    data[c] = data[c].fillna(data[c].median())

sample_NA_median = data.loc[50, ['CreditScore', 'Age']]

#Comparing different imputing methods
sample = pd.concat([sample_NA, sample_NA_mean, sample_NA_median], axis = 1)

sample.columns = ['sample_NA', 'sample_NA_mean', 'sample_NA_median']
 
#Inserting a new blank row in "sample"
sample['Best Method'] = np.nan

#Finding the best method of imputation for each column
for d in range(sample.shape[0]):
    if  (abs(sample.iloc[d, 0]-sample.iloc[d, 1]) < abs(sample.iloc[d, 0]-sample.iloc[d, 2])):
        sample.iloc[d, 3] = "MEAN"
    else:
        sample.iloc[d, 3] = "MEDIAN"


#From "sample" dataframe we can find the best method for each column





#**************************************************************************************

#Imputing the best fit method for each column
#Re-Run the data till-"ONLY FOR FINDING THE BEST METHOD"
data['CreditScore'] = data['CreditScore'].fillna(data['CreditScore'].median())   
data['Age'] = data['Age'].fillna(data['Age'].mean())





#***********************************DATA MANIPULATION***********************************

#Re-run the data upto line 54

#Deleting unnecessary variables
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)





#***********************************EXPOLATORY DATA ANALYSIS AND FEATURE SELECTION***********************************

#Correlation analysis
import matplotlib.pyplot as plt
num_cnames = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
data_corr = data.loc[:, num_cnames]

#Set the height and width of the plot
f,ax = plt.subplots(figsize = (7,5))

#Generate correlation matrix
corr = data_corr.corr()

#plot using seaborn
import seaborn as sns
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool),
            cmap = sns.diverging_palette(220, 10, as_cmap = True),
            square = True, ax = ax)

#From the correlation plot we can see that there is no high correlation between numerical variables



#Now to check the correlation between the categorical variables we would go for Chi_sq test

cat_names = ["Geography", "Gender", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember"]

from scipy.stats import chi2_contingency

for j in cat_names:
    print(j)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(data['Exited'], data[j]))
    print(p)

#The p_value of "Tenure" and "HasCrCard" is greater than 0.05
#So we can see that "Tenure" and "HasCrCard" doesn't help to explain the variance of the dependent variable
#Thus we will delete them from further consideration

data = data.drop(["Tenure", "HasCrCard"], axis = 1)



#Now to check the relationship of numerical variables with the dependent variable we will opt for ANOVA Test

#ANOVA(Analysis of Variance) Test
import statsmodels.api as sm
from statsmodels.formula.api import ols

#ANOVA for CreditScore
data.boxplot('CreditScore' , by = 'Exited')
Exited_CreditScore_ols = ols('CreditScore ~ Exited', data = data).fit()
sm.stats.anova_lm(Exited_CreditScore_ols, type = 1)

#So from the plot we can see that the mean of Credit Score is almost same across groups
#As p_value is greater than 0.05 so we can say that CreditScore doesn't explain the variance of the dependent variable



#ANOVA for Age
data.boxplot('Age' , by = 'Exited')
Exited_Age_ols = ols('Age ~ Exited', data = data).fit()
sm.stats.anova_lm(Exited_Age_ols, type = 1)

#So from the plot we can see that the mean of Age is different across groups
#As p_value less than 0.05 so we can say that Exited is dependent on Age



#ANOVA for Balance
data.boxplot('Balance' , by = 'Exited')
Exited_Balance_ols = ols('Balance ~ Exited', data = data).fit()
sm.stats.anova_lm(Exited_Balance_ols, type = 1)

#So from the plot we can see that the mean of Balance is different across groups
#As p_value less than 0.05 so we can say that Exited is dependent on Balance



#ANOVA for Estimated Salary
data.boxplot('EstimatedSalary' , by = 'Exited')
Exited_EstimatedSalary_ols = ols('EstimatedSalary ~ Exited', data = data).fit()
sm.stats.anova_lm(Exited_EstimatedSalary_ols, type = 1)

#So from the plot we can see that the mean of Estimated Salary is almost same across the groups
#p_value being greater than 0.05 so we can say that this is not a good predictor pf our dependent variable



#Thus we will delete "CreditScore" and "EstimatedSalary" for further consideration
data = data.drop(["CreditScore", "EstimatedSalary"], axis = 1)





#***********************************FEATURE SCALING***********************************

#Checking the distribution of the numerical variables for choosing the scaling method

#Age
sns.distplot(data['Age'])
#Histogram plot is close to a normal distribution

#Balance
sns.distplot(data['Balance'])
#The histogram is very close to normal distribution

#Normalization would be applied to scale as the all the variables are not normally distributed



import numpy as np

cnames = ["Age", "Balance"]

for k in cnames:
    data[k] = ((data[k] - np.min(data[k])) / (np.max(data[k]) - np.min(data[k])))





#***********************************TRAIN-TEST SPLIT***********************************
#Checking the number of values in each factor variable
data['Exited'].value_counts()                   #Slightly Biased
data['Geography'].value_counts()                #almost equal across groups
data['Gender'].value_counts()                   #almost equally distributed
data['NumOfProducts'].value_counts()            #Highly biased
data['IsActiveMember'].value_counts()           #almost equal across groups

#So as we can see that the dataset is baised across some categories so if we apply random sampling in this case
#Then there might be a chance that no observations of the low count groups is included
#So we need to apply stratified spliting in this case taking the most correlated variable as reference variable

#From the correlation plot we can see that Number of Products would be the best variable to create the strata as it is highly biased and good correlation with dependent variable
np.random.seed(555)

from sklearn.model_selection import train_test_split
#Categorical variable to be set as an array
y = np.array(data['NumOfProducts'])
training_set,test_set = train_test_split(data, test_size = 0.2, stratify = y)





#***********************************ENCODING CATEGORICAL VARIABLES***********************************

#For Training Set
dummies_train = pd.get_dummies(training_set[['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember']], drop_first = True)
training_set = pd.concat([dummies_train, training_set[['Age', 'Balance', 'Exited']]], axis = 1)

#For Test Set
dummies_test = pd.get_dummies(test_set[['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember']], drop_first = True)
test_set = pd.concat([dummies_test, test_set[['Age', 'Balance', 'Exited']]], axis = 1)





#***********************************MODEL BUILDING***********************************

#Logistic Regression
from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression(random_state = 0).fit(training_set.iloc[:, 0:9], training_set['Exited'])
LR_predictions = LR_model.predict(test_set.iloc[:, 0:9])

from sklearn.metrics import confusion_matrix, accuracy_score
cm_LR = confusion_matrix(test_set.iloc[:, 9], LR_predictions)
accuracy_LR = accuracy_score(test_set.iloc[:, 9], LR_predictions)



#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
DT_model = DecisionTreeClassifier(criterion = 'entropy').fit(training_set.iloc[:, 0:9], training_set['Exited'])
DT_predictions = DT_model.predict(test_set.iloc[:, 0:9])

cm_DT = confusion_matrix(test_set.iloc[:, 9], DT_predictions)
accuracy_DT = accuracy_score(test_set.iloc[:, 9], DT_predictions)



#Random Forest
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 100).fit(training_set.iloc[:, 0:9], training_set['Exited'])
RF_predictions = RF_model.predict(test_set.iloc[:, 0:9])

cm_RF = confusion_matrix(test_set.iloc[:, 9], RF_predictions)
accuracy_RF = accuracy_score(test_set.iloc[:, 9], RF_predictions)



#K Nearest Neighbors(KNN)
from sklearn.neighbors import KNeighborsClassifier
KNN_model = KNeighborsClassifier(n_neighbors = 1).fit(training_set.iloc[:, 0:9], training_set['Exited'])
KNN_predictions = KNN_model.predict(test_set.iloc[:, 0:9])

cm_KNN = confusion_matrix(test_set.iloc[:, 9], KNN_predictions)
accuracy_KNN = accuracy_score(test_set.iloc[:, 9], KNN_predictions)



#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB_model = GaussianNB().fit(training_set.iloc[:, 0:9], training_set['Exited'])
NB_predictions = NB_model.predict(test_set.iloc[:, 0:9])

cm_NB = confusion_matrix(test_set.iloc[:, 9], NB_predictions)
accuracy_NB = accuracy_score(test_set.iloc[:, 9], NB_predictions)



#SVM(Support Vector Machine)
from sklearn.svm import SVC
SVC_model = SVC(kernel = 'rbf', random_state = 0).fit(training_set.iloc[:, 0:9], training_set['Exited'])
SVC_predictions = SVC_model.predict(test_set.iloc[:, 0:9])

cm_SVC = confusion_matrix(test_set.iloc[:, 9], SVC_predictions)
accuracy_SVC = accuracy_score(test_set.iloc[:, 9], SVC_predictions)



#So we can see that all the models gives accuracy greater than 80%
#So we can say that all the models gives good prediction
#The accuracy is highest for Logistic Regression Model
#But as the problem is customer churn detection, so we have to look at he false negative rate as well
#Because our model should be able to detect those customers who are going to leave so that we can take precautions
#If our model predicts that a customer is not going to leave and he is actually leaving so that is the worst condition
#Considering this we can select Random Forest as the best model






















