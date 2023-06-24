library(dplyr)
library(mice)
library(randomForest)
library(ggplot2)
library(rpart)
library(imbalance)
library(randomForest)
library(tidyverse)
library(pROC)
library(caret)
library(cluster) # for gower similarity and pam
library(Rtsne) # for t-SNE plot
library(ggplot2) # for visualization
library(philentropy)
library(BBmisc)
library(factoextra)

#Load dataset
data_train = read.csv("C:\\Users\\daoto\\Documents\\adult\\adult.csv")
data_test = read.csv("C:\\Users\\daoto\\Documents\\adult\\adult_test.csv")
combined_table <- rbind(data_train, data_test)
features <- colnames(combined_table)

####Data preprocessing####
#Clean null value
NA_data <- replace(combined_table, combined_table == " ?", NA)

na_row <- sum(!complete.cases(NA_data))
print(na_row) #3620 rows with missing value

#removed NA data
rNA_data = na.omit(NA_data)

#Transform imp_data
NA_data$workclass <- as.factor(NA_data$workclass)
NA_data$education <- as.factor(NA_data$education)
NA_data$marital_status <- as.factor(NA_data$marital_status)
NA_data$occupation <- as.factor(NA_data$occupation)
NA_data$relationship <- as.factor(NA_data$relationship)
NA_data$race <- as.factor(NA_data$race)
NA_data$sex <- as.factor(NA_data$sex)
NA_data$native_country <- as.factor(NA_data$native_country)
NA_data$income_bracket <- as.factor(ifelse(NA_data$income_bracket == " >50K","above", "under"))



#Transform nonimp data
rNA_data$workclass <- as.factor(rNA_data$workclass)
rNA_data$education <- as.factor(rNA_data$education)
rNA_data$marital_status <- as.factor(rNA_data$marital_status)
rNA_data$occupation <- as.factor(rNA_data$occupation)
rNA_data$relationship <- as.factor(rNA_data$relationship)
rNA_data$race <- as.factor(rNA_data$race)
rNA_data$sex <- as.factor(rNA_data$sex)
rNA_data$native_country <- as.factor(rNA_data$native_country)
rNA_data$income_bracket <- as.factor(ifelse(rNA_data$income_bracket == " >50K","above", "under"))

###data imputation
#using mice random forest
set.seed(7)
mrf <- NA_data 
impute_rf <- function(data) {
  
  imp <- mice(data, method = "rf")
  data <- complete(imp)
  
  return(data)
}

imp_data <- impute_rf(mrf)

###normalizing data
cont_cols <- c("age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week", "education_num") 
s_imp_data <- imp_data
s_imp_data[, cont_cols] <- scale(s_imp_data[, cont_cols])
s_rNA_data<-  rNA_data
s_rNA_data[, cont_cols] <- scale(s_rNA_data[, cont_cols])


###Outliers analysis and data distribution
par(mfrow = c(2, 3))
boxplot(combined_table$age, main = "Age")
boxplot(combined_table$fnlwgt, main="final_weight")
boxplot(combined_table$hours_per_week, main="hours_per_week")
boxplot(combined_table$capital_gain, main="capital_gain")
boxplot(combined_table$education_num, main="education_num")
boxplot(combined_table$capital_loss, main="capital_loss")

ggplot(data = rNA_data, aes(x = age)) +
  geom_density() + labs(x = "age", y = "Density", title = "Distribution of age", alpha = 0.3)
ggplot(data = rNA_data, aes(x = fnlwgt)) +
  geom_density() + labs(x = "final weight", y = "Density", title = "Distribution of final weight")
ggplot(data = rNA_data, aes(x = education_num)) +
  geom_density() + labs(x = "education duration", y = "Density", title = "Distribution of education duration")
ggplot(data = rNA_data, aes(x = capital_gain)) +
  geom_density() + labs(x = "capital_gain", y = "Density", title = "Distribution of capital_gain")
ggplot(data = rNA_data, aes(x = capital_loss)) +
  geom_density() + labs(x = "capital_loss", y = "Density", title = "Distribution of capital_loss")
ggplot(data = rNA_data, aes(x = hours_per_week)) +
  geom_density() + labs(x = "hours per week", y = "Density", title = "Distribution of hours per week")

ggplot(data = rNA_data, aes(x = workclass)) +
  geom_bar() +
  labs(x = "Workclass", y = "Count", title = "Distribution of workclass")
ggplot(data = rNA_data, aes(x = education)) +
  geom_bar() +
  labs(x = "education", y = "Count", title = "Distribution of education")
ggplot(data = rNA_data, aes(x = marital_status)) +
  geom_bar() +
  labs(x = "marital_status", y = "Count", title = "Distribution of marital status")

ggplot(data = rNA_data, aes(x = relationship)) +
  geom_bar() +
  labs(x = "relationship", y = "Count", title = "Distribution of relationship")
ggplot(data = rNA_data, aes(x = race)) +
  geom_bar() +
  labs(x = "race", y = "Count", title = "Distribution of race")
ggplot(data = rNA_data, aes(x = sex)) +
  geom_bar() +
  labs(x = "sex", y = "Count", title = "Distribution of sex")



# calculate number of outliers
count_outliers <- function(data, column, threshold = 1.5) {

  q1 <- quantile(data[[column]], 0.25)
  q3 <- quantile(data[[column]], 0.75)
  iqr <- q3 - q1
  
  lower_bound <- q1 - threshold * iqr
  upper_bound <- q3 + threshold * iqr
  
  outliers <- data[[column]][data[[column]] < lower_bound | data[[column]] > upper_bound]
  
  num_outliers <- length(outliers)
  
  return(num_outliers)
}
num_outliers <- count_outliers(combined_table, "workclass")
print(num_outliers)




###Imbalances in feature, so over sampling
set.seed(7)
training = downSample(s_rNA_data, s_rNA_data$sex, list = FALSE)
training = training[-16]

training_up = upSample(s_rNA_data, s_rNA_data$sex, list = FALSE)
training_up = training[-16]


ggplot(data = training, aes(x = sex)) +
  geom_bar() +
  labs(x = "sex", y = "Count", title = "Distribution of sex")




###Building Tree###
#with non imp data, imb data
partition_data <- createDataPartition(s_rNA_data$income_bracket, p = 0.8, list = FALSE)
training_tree <- s_rNA_data[partition_data, ]
testing_tree <- s_rNA_data[-partition_data, ]
train_tree<-rpart(income_bracket~. -income_bracket, data=training_tree)
treerf = rpart.plot::prp(train_tree, digits = 5)
x_test<-testing_tree[, 1:14]
y_test<-testing_tree[,15]
test_test_rf <- predict(train_tree, x_test, type = "class")
cf <- confusionMatrix(data=factor(test_test_rf), reference = y_test)
cf
## roc curves ##
y_test = ifelse(y_test == "above", 0, 1)
test_test_rf = ifelse(test_test_rf == "above", 1, 0)
roc.rf = roc(y_test, test_test_rf)
plot.roc(roc.rf)

#with imp data, imb data
partition_data <- createDataPartition(s_imp_data$income_bracket, p = 0.8, list = FALSE)
training_tree <- s_imp_data[partition_data, ]
testing_tree <- s_imp_data[-partition_data, ]
train_tree<-rpart(income_bracket~. -income_bracket, data=training_tree)
treerf = rpart.plot::prp(train_tree, digits = 4)
x_test<-testing_tree[, 1:14]
y_test<-testing_tree[,15]
test_test_rf <- predict(train_tree, x_test, type = "class")
cf <- confusionMatrix(data=factor(test_test_rf), reference = y_test)
cf
## roc curves ##
y_test = ifelse(y_test == "above", 0, 1)
test_test_rf = ifelse(test_test_rf == "above", 1, 0)
roc.rf = roc(y_test, test_test_rf)
plot.roc(roc.rf)

#with non imp data, balance data
partition_data <- createDataPartition(training$income_bracket, p = 0.8, list = FALSE)
training_tree <- training[partition_data, ]
testing_tree <- training[-partition_data, ]
train_tree<-rpart(income_bracket~. -income_bracket, data=training_tree)
treerf = rpart.plot::prp(train_tree, digits = 5)
x_test<-testing_tree[, 1:14]
y_test<-testing_tree[,15]
test_test_rf <- predict(train_tree, x_test, type = "class")
cf <- confusionMatrix(data=factor(test_test_rf), reference = y_test)
cf
## roc curves ##
y_test = ifelse(y_test == "above", 0, 1)
test_test_rf = ifelse(test_test_rf == "above", 1, 0)
roc.rf = roc(y_test, test_test_rf)
plot.roc(roc.rf)

var_importance <- importance(train_tree, type = 1)
plot(var_importance, main = "Variable Importance Plot (Mean Decreased Gini)")
###Random Forest###
#1 no imp, imb data
train <- sample(1: nrow (s_rNA_data), nrow (s_rNA_data)*0.80)
rt <- randomForest(income_bracket~ ., data = s_rNA_data, subset = train, importance = TRUE)

y.test.rt <- s_rNA_data[-train, "income_bracket"]
yhat.rt <- predict (rt , newdata = s_rNA_data[-train , ])
plot(yhat.rt, y.test.rt)

confusionMatrix(yhat.rt, y.test.rt) 

y.test.rt = ifelse(y.test.rt == "above", 1,0)
yhat.rt = ifelse(yhat.rt == "above", 1,0)
roc.rff = roc(y.test.rt, yhat.rt)
plot.roc(roc.rff, col = "green", add = FALSE, grid = TRUE)

var_importance <- importance(rt, type = 1)
plot(var_importance, main = "Variable Importance Plot (Mean Decreased Gini)")
#2 imp, imb data
train <- sample(1: nrow (s_imp_data), nrow (s_imp_data)*0.80)
rt <- randomForest(income_bracket~ ., data = s_imp_data, subset = train, importance = TRUE)

y.test.rt <- s_imp_data[-train, "income_bracket"]
yhat.rt <- predict (rt , newdata = s_imp_data[-train , ])
plot(yhat.rt, y.test.rt)

confusionMatrix(yhat.rt, y.test.rt) 

y.test.rt = ifelse(y.test.rt == "above", 1,0)
yhat.rt = ifelse(yhat.rt == "above", 1,0)
roc.rff = roc(y.test.rt, yhat.rt)
plot.roc(roc.rff, col = "green", add = FALSE, grid = TRUE)

#3 no imp, balance data
train <- sample(1: nrow (training), nrow (training)*0.80)
rt <- randomForest(income_bracket~ ., data = training, subset = train,  importance = TRUE)

y.test.rt <-training[-train, "income_bracket"]
yhat.rt <- predict (rt , newdata = training[-train , ])
plot(yhat.rt, y.test.rt)

confusionMatrix(yhat.rt, y.test.rt) 

y.test.rt = ifelse(y.test.rt == "above", 1,0)
yhat.rt = ifelse(yhat.rt == "above", 1,0)
roc.rff = roc(y.test.rt, yhat.rt)
plot.roc(roc.rff, col = "green", add = FALSE, grid = TRUE)

var_importance <- importance(rt, type = 1)
plot(var_importance, main = "Variable Importance Plot (Mean Decreased Gini)")
###bagging###
#1 no imp, imbalance data
train <- sample(1: nrow (s_rNA_data), nrow (s_rNA_data)*0.80)
rt <- randomForest(income_bracket~ ., data = s_rNA_data , subset = train, mtry = 6,  importance = TRUE)

y.test.rt <-s_rNA_data[-train, "income_bracket"]
yhat.rt <- predict (rt , newdata = s_rNA_data[-train , ])
plot(yhat.rt, y.test.rt)

confusionMatrix(yhat.rt, y.test.rt) 

y.test.rt = ifelse(y.test.rt == "above", 1,0)
yhat.rt = ifelse(yhat.rt == "above", 1,0)
roc.rff = roc(y.test.rt, yhat.rt)
plot.roc(roc.rff, col = "green", add = FALSE, grid = TRUE)
sorted_var_importance <- sort(var_importance, decreasing = TRUE)


var_importance <- importance(rt, type = 1)
plot(var_importance, main = "Variable Importance Plot (Mean Decreased Gini)")

# imp, imbalance data
train <- sample(1: nrow (s_imp_data), nrow (s_imp_data)*0.80)
rt <- randomForest(income_bracket~ ., data = s_imp_data, subset = train, mtry = 8 ,  importance = TRUE)

y.test.rt <-s_imp_data[-train, "income_bracket"]
yhat.rt <- predict (rt , newdata = s_imp_data[-train , ])
plot(yhat.rt, y.test.rt)

confusionMatrix(yhat.rt, y.test.rt) 

y.test.rt = ifelse(y.test.rt == "above", 1,0)
yhat.rt = ifelse(yhat.rt == "above", 1,0)
roc.rff = roc(y.test.rt, yhat.rt)
plot.roc(roc.rff, col = "green", add = FALSE, grid = TRUE)


# no imp, balance data
train <- sample(1: nrow (training), nrow (training)*0.80)
rt <- randomForest(income_bracket~ ., data =training , subset = train, mtry = 8,  importance = TRUE)

y.test.rt <-training[-train, "income_bracket"]
yhat.rt <- predict (rt , newdata = training[-train , ])
plot(yhat.rt, y.test.rt)

confusionMatrix(yhat.rt, y.test.rt) 

y.test.rt = ifelse(y.test.rt == "above", 1,0)
yhat.rt = ifelse(yhat.rt == "above", 1,0)
roc.rff = roc(y.test.rt, yhat.rt)
plot.roc(roc.rff, col = "green", add = FALSE, grid = TRUE)

var_importance <- importance(rt, type = 1)
plot(var_importance, main = "Variable Importance Plot (Mean Decreased Gini)")



### unsupervised###
num <- rNA_data[c("age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week", "education_num") ]


norm = normalize(num, method = "standardize", range = c(0, 1), margin = 1L, on.constant = "quiet")

hc.c = select(rNA_data, subset = -c("age", "fnlwgt", "capital_gain", "capital_loss", "hours_per_week", "education_num"))
hc.c = cbind(hc.c, norm)
head(hc.c)
total_rows <- nrow(hc.c)

new_size <- total_rows %/% 20

hcc <- hc.c[sample(total_rows, new_size), ]
#measure gower distance no imp, imb
g_d <- daisy(hcc,
                    metric = "gower",
                    type = list(logratio = 3))

## building cluster ###

hc.av <- hclust(g_d, method = "average")
plot(hc.av)
plot(hc.av, abline(h=15, col="red", xlab = "The clusters have been created using \"complete\" linkage method"))

groups <- cutree(hc.av, k = 2)

table(groups, hcc$income_bracket)

clusplot(hcc, groups, main='2D representation of the Cluster solution',
         color=TRUE, shade=TRUE,
         labels=1, lines=0)

#under
total_rows <- nrow(training)

new_size <- total_rows %/% 13

hcc <- training[sample(total_rows, new_size), ]
#measure gower distance no imp, imb
g_d <- daisy(hcc,
             metric = "gower",
             type = list(logratio = 3))

## building cluster ###

hc.av <- hclust(g_d, method = "average")
plot(hc.av)
plot(hc.av, abline(h=15, col="red", xlab = "The clusters have been created using \average\" linkage method"))

groups <- cutree(hc.av, k = 2)

table(groups, hcc$income_bracket)

clusplot(hcc, groups, main='2D representation of the Cluster solution',
         color=TRUE, shade=TRUE,
         labels=1, lines=0)