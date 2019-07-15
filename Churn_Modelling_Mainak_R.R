#Churn Modelling

#Clearing the workspace
rm(list = ls())

#Setting the working directory
setwd("C:/Users/Mainak Sarkar/Desktop/Kaggle/Bank_Customer_Churn_Detection")

#Importing the data skipping the blank lines if any
data = read.csv("Churn_Modelling.csv", blank.lines.skip = TRUE)

#Checking the summary of the whole data set
summary(data)

#Checking class of each column
class(data$RowNumber)                              #integer
class(data$CustomerId)                             #integer
class(data$Surname)                                #factor
class(data$CreditScore)                            #integer
class(data$Geography)                              #factor
class(data$Gender)                                 #factor
class(data$Age)                                    #integer
class(data$Tenure)                                 #integer
class(data$Balance)                                #numeric
class(data$NumOfProducts)                          #integer
class(data$HasCrCard)                              #integer
class(data$IsActiveMember)                         #integer
class(data$EstimatedSalary)                        #numeric
class(data$Exited)                                 #integer





#***********************************DATA PREPROCESSING***********************************

#Data type conversion

#converting "Tenure" as factor
data$Tenure = as.factor(data$Tenure)

#converting "NumOfProducts" as factor
data$NumOfProducts = as.factor(data$NumOfProducts)

#converting "HasCrCard" as factor
data$HasCrCard = as.factor(data$HasCrCard)

#converting "IsActiveMember" as factor
data$IsActiveMember = as.factor(data$IsActiveMember)

#converting "Tenure" as factor
data$Exited = as.factor(data$Exited)





#***********************************MISSING VALUE ANALYSIS***********************************

#Checking null values
missing_value = data.frame(apply(data, 2, function(x){sum(is.na(x))}))

#Calculating Percentage of missing values
missing_value$percentage = (missing_value[, 1]/nrow(data))*100

#SO the dataset has no missing values





#***********************************OUTLIER ANALYSIS***********************************

#Selecting numerical variables
numeric_index = sapply(data, is.numeric)
numeric_data = data[, numeric_index]
numerical_cnames = colnames(numeric_data)

#Imputing null values in place of outliers
for (i in numerical_cnames[-c(1,2)])                    #"instant" and all dependent variables are removed
{
  val = data[,i][data[,i]%in%boxplot.stats(data[,i])$out]
  data[,i][data[,i]%in%val] = NA
}

#Checking number of outliers(null values)
outliers = data.frame(apply(data, 2, function(y){sum(is.na(y))}))

#Calculating Percentage of outliers(null values)
outliers$percentage = (outliers[,1]/nrow(data))*100

#As we know that presence of ouliers can affect our models a lot
#So we either have to delete them or go for imputation
#But as the no. of outliers is high so we will opt for imputation





#***********************************ONLY FOR FINDING THE BEST METHOD***********************************

#Selecting variables containing NAs
NA_index = sapply(data, anyNA)
NA_data = data[, NA_index]
NA_cnames = colnames(NA_data)

#Choosing the best method for missing value imputation

#Making a sample to check which method works best
#Choosing a sample and saving its value
sample_NA = data[51, c(4,7)]

#Putting values of sample equal to NA for required columns
data[51,c(NA_cnames)] = NA

#duplicating data
data_duplicate = data

#MEAN Method
for(b in NA_cnames)
  data[, b][is.na(data[, b])] = mean(data[, b], na.rm = TRUE)

sample_NA_mean = data[51, c(4,7)]

#MEDIAN Method
data = data_duplicate
for(c in NA_cnames)
  data[, c][is.na(data[, c])] = median(data[, c], na.rm = TRUE)

sample_NA_median = data[51, c(4,7)]

#Comparing different imputing methods
sample = rbind(sample_NA, sample_NA_mean, sample_NA_median)

#Inserting a new blank row in "sample"
sample[nrow(sample)+1, ]=NA

#Changing row names
row.names(sample) = c("sample_NA","sample_NA_mean","sample_NA_median","Best Method")

#Finding the best method of imputation for each column
for (d in (1:ncol(sample)))
{
  if(abs(as.numeric(sample[1,d])-as.numeric(sample[2,d]))<abs(as.numeric(sample[1,d])-as.numeric(sample[3,d])))
  {
    sample[4,d] = "MEAN"
  } else {
    sample[4,d] = "MEDIAN"
  }
}

#From "sample" dataframe we can find the best method for each column





#**************************************************************************************

#Imputing the best fit method for each column
#Re-Run the data till-"ONLY FOR FINDING THE BEST METHOD"
data$CreditScore[is.na(data$CreditScore)] = median(data$CreditScore, na.rm = TRUE)
data$Age[is.na(data$Age)] = mean(data$Age, na.rm = TRUE)





#***********************************DATA MANIPULATION***********************************

#Deleting unnecessary variables
data = data[, -c(1,2,3)]





#***********************************EXPOLATORY DATA ANALYSIS AND FEATURE SELECTION***********************************

#Correlation Analysis
cor_index = sapply(data, is.numeric)

library(corrgram)
corrgram(data[, cor_index], order = FALSE, upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation Plot")

#From the correlation plot we can see that there is no high correlation between numerical variables



#Now to check the correlation between the categorical variables we would go for Chi_sq test

factor_index = sapply(data, is.factor)
factor_data = data[, factor_index]

for(k in (1:ncol(factor_data)))
{
  print(names(factor_data)[k])
  print(chisq.test(table(factor_data$Exited, factor_data[, k])))
}

#The p_value of "Tenure" and "HasCrCard" is greater than 0.05
#So we can see that "Tenure" and "HasCrCard" doesn't help to explain the variance of the dependent variable
#Thus we will delete them from further consideration

data = data[, -c(5,8)]



#Now to check the relationship of numerical variables with the dependent variable we will opt for ANOVA Test

#ANOVA(Analysis of Variance) Test

#ANOVA for CreditScore
plot(CreditScore ~ Exited, data = data)
#So from the plot we can see that the mean of Credit Score is almost same across groups

#Checking the p_value
summary(aov(CreditScore ~ Exited, data = data))
#As p_value is greater than 0.05 so we can say that CreditScore doesn't explain the variance of the dependent variable



#ANOVA for Age
plot(Age ~ Exited, data = data)
#So from the plot we can see that the mean of Age is different across groups

#Checking the p_value
summary(aov(Age ~ Exited, data = data))
#As p_value less than 0.05 so we can say that Exited is dependent on Age



#ANOVA for Balance
plot(Balance ~ Exited, data = data)
#So from the plot we can see that the mean of Balance is different across groups

#Checking the p_value
summary(aov(Balance ~ Exited, data = data))
#As p_value less than 0.05 so we can say that Exited is dependent on Balance



#ANOVA for Estimated Salary
plot(EstimatedSalary ~ Exited, data = data)
#So from the plot we can see that the mean of Estimated Salary is almost same across the groups

#Checking the p_value
summary(aov(EstimatedSalary ~ Exited, data = data))
#p_value being greater than 0.05 so we can say that this is not a good predictor pf our dependent variable



#Thus we will delete "CreditScore" and "EstimatedSalary" for further consideration
data = data[, -c(1,8)]





#***********************************FEATURE SCALING***********************************

#Checking the distribution of the numerical variables for choosing the scaling method
#Age
qqnorm(data$Age)
qqline(data$Age)
#The q plot deviates from the q line a lot
hist(data$Age)
#Histogram plot is close to a normal distribution



#Balance
qqnorm(data$Balance)
qqline(data$Balance)
#The q plot doesn't follow the q line at all
hist(data$Balance)
#Histogram plot is not normally distributed



#Normalization would be applied to scale as the all the variables are not normally distributed



num_cnames = c("Age", "Balance")
for (j in num_cnames)
 {
   data[, j] = (data[, j] - min(data[, j])) / (max(data[, j]) - min(data[, j]))
 }


#Renaming the categories
library(plyr)
data$Geography = revalue(data$Geography, c("France" = "1", "Spain" = "2", "Germany" = "3"))
data$Gender = revalue(data$Gender, c("Female" = "1", "Male" = "2"))



#***********************************TRAIN-TEST SPLIT***********************************

table(data$Exited)                          #Slightly baised
table(data$Geography)                       #Almost equal among groups
table(data$Gender)                          #Almost Equal among groups
table(data$NumOfProducts)                   #Highly Biased
table(data$IsActiveMember)                  #Almost Equal among groups

#So as we can see that the dataset is baised across some categories so if we apply random sampling in this case
#Then there might be a chance that no observations of the low count groups is included
#So we need to apply stratified spliting in this case taking the most correlated variable as reference variable

#From the correlation plot we can see that Number of Products would be the best variable to create the strata as it is highly biased and good correlation with dependent variable
set.seed(123)
library(caret)
train.index = createDataPartition(data$NumOfProducts, p = 0.8, list = FALSE)
training_set = data[train.index,]
test_set = data[-train.index,]





#***********************************MODEL BUILDING***********************************

#Logistic Regression
LR_model = glm(Exited ~ ., data = training_set, family = "binomial")
summary(LR_model)
Logit_predictions = predict(LR_model, newdata = test_set[, -7], type = "response")
LR_predictions = ifelse(Logit_predictions > 0.5, 1, 0)

conf_matrix_LR = table(test_set$Exited, LR_predictions)
confusionMatrix(conf_matrix_LR)



#Decision Tree Classifier
library(C50)
C50_model = C5.0(Exited ~ ., training_set, trials = 100, rules = TRUE)
C50_predictions = predict(C50_model, test_set[, -7], type = "class")

conf_matrix_C50 = table(test_set$Exited, C50_predictions)
confusionMatrix(conf_matrix_C50)



#Random Forest
library(randomForest)
RF_model = randomForest(Exited ~ ., training_set, ntree = 500)
RF_predictions = predict(RF_model, test_set[, -7], type = "class")

conf_matrix_RF = table(test_set$Exited, RF_predictions)
confusionMatrix(conf_matrix_RF)



#K Nearest Neighbors(KNN)
library(class)
KNN_predictions = knn(training_set[, 1:6], test_set[, 1:6], training_set$Exited, k = 3)

conf_matrix_KNN = table(test_set$Exited, KNN_predictions)
confusionMatrix(conf_matrix_KNN)

#As the dependent variable is categorical so we have imputed only the odd values ok
#K = 3 gives the best value



#Naive Bayes
library(e1071)
NB_model = naiveBayes(Exited ~ ., data = training_set)
NB_predictions = predict(NB_model, test_set[, 1:6], type = "class")

conf_matrix_NB = table(test_set$Exited, NB_predictions)
confusionMatrix(conf_matrix_NB)



#SVM(Support Vector Machine)
library(e1071)
SVM_model = svm(formula = Exited ~ .,
                data = training_set,
                type = "C-classification",
                kernel = "linear")
SVM_predictions = predict(SVM_model, type = "response", newdata = test_set[, -7])

conf_matrix_SVM = table(test_set$Exited, SVM_predictions)
confusionMatrix(conf_matrix_SVM)



#ANN
library(h2o)
h2o.init(nthreads = -1)
ANN_model = h2o.deeplearning(y = "Exited",
                             training_frame = as.h2o(training_set),
                             activation = "Rectifier",
                             hidden = c(4,4),
                             epochs = 100,
                             train_samples_per_iteration = -2)
prod_pred = as.data.frame(h2o.predict(ANN_model, newdata = as.h2o(test_set[, -7])))
ANN_predictions = prod_pred[, 1]

conf_matrix_ANN = table(test_set$Exited, ANN_predictions)
confusionMatrix(conf_matrix_ANN)

h2o.shutdown()
Y



#So we can see that all the models gives accuracy greater than 80%
#So we can say that all the models gives good prediction
#The accuracy is highest for Decision Tree Model
#But as the problem is customer churn detection, so we have to look at he false negative rate as well
#Because our model should be able to detect those customers who are going to leave so that we can take precautions
#If our model predicts that a customer is not going to leave and he is actually leaving so that is the worst condition
#Considering this we can select ANN as the best model


