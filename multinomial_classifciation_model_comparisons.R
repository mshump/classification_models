

############################
# Install Libraries (1 time)
###########################


#install.packages("VGAM")
#install.packages("caret")
#install.packages("pbkrtest")
#install.packages("e1071")
#install.packages("caret")
#install.packages("mda")
#install.packages("MASS")
#install.packages("klaR")
#install.packages("nnet")
#install.packages("kernlab")
#install.packages("rpart")
#install.packages("RWeka")
#install.packages("ipred")
#install.packages("randomForest")
#install.packages("gbm")
#install.packages("C50")
#install.packages("plotly")
#install.packages(reshape2)
#install.packages(ggplot2)

############################
# Load Libraries
###########################

library(C50)
library(gbm)
library(randomForest)
library(ipred)
library(RWeka)
library(rpart)
library(kernlab)
library(nnet)
library(klaR)
library(mda)
library(caret)
library(VGAM)
library(MASS)
library(e1071)
library(plotly)
library(reshape2)
library(ggplot2)

#############
# Get the data
#############

# https://github.com/surajvv12/17_Classification/blob/master/Data/car.data.txt

#data_in <- read.csv("C:/Users/Matt/Documents/Projects/R Scripts/17 classifications/car_data.csv", header=FALSE)
#colnames(data_in)<-c("buying","maint","doors","persons","lug_boot","safety","class")

#install.packages("mlbench")
library(mlbench)
data(Vehicle)
str(Vehicle)
data_in <- Vehicle
#data(Vehicle)
#head(Vehicle)
#summary(Vehicle)


#############
# Basic summaries?
#############

summary(data_in)
str(data_in)


#######################################
# Identify the predictor and response variables
######################################
#colnames(data_in) <- tolower(colnames(data_in))
resp_var <- c("Class") #class
pred_vars <- unlist(colnames(data_in[,!(colnames(data_in) %in% resp_var)]))
#pred_vars <- c("buying","maint","doors","persons","lug_boot","safety")


######################################
# Make formula function
######################################

Formula <- formula(paste(resp_var,"~ ",  paste(pred_vars, collapse=" + ")))

fact_labels <-levels(eval(parse(text=paste0('data_in$',resp_var))))
fact_levels <- seq(1,length(fact_labels), 1)


##############################
# STEP 1: Test performance
#############################

######################################
# Split data to test and training sets
######################################
# within population , not based on future time period
# set.seed(3456)

train_ratio <- 0.75       #define % of training and test set
trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      

data_in_train <- data_in[trainindex, ]   #get training set
data_in_train$data_set <- 'train' 
data_in_test <- data_in[-trainindex, ]     #get test set                                   
data_in_test$data_set <- 'test'

# Hows are response distribution between data sets?
# Want to be similar
prev_train <- table(eval(parse(text=paste0('data_in_train$',resp_var))))
prop.table(table(eval(parse(text=paste0('data_in_train$',resp_var)))))
prev_test <- table(eval(parse(text=paste0('data_in_test$',resp_var))))
prop.table(table(eval(parse(text=paste0('data_in_test$',resp_var)))))

# quick test on difference in response prevalance rates
chi2 <- chisq.test( rbind(prev_train,prev_test)) 
chi2$p.value  



######################################
# Make in / out sets for train and test
######################################
str(data_in)
str(data_in_train)
pred_vars
x_train <- data_in_train[,pred_vars]
#y_train <- data_in_train[,resp_var]

x_test <- data_in_test[,pred_vars]
#y_test <- data_in_test[,resp_var]


######################################
# Set up model meta data table
######################################

model_meta <- data.frame(model_id= integer(0)
                         , model_name= character(0)
                         , model_alias= character(0)
                         , req_pack= character(0)
                         , func = character(0)
                         , mod_arg = character(0))

######################################
# Begin Models
######################################

#Logistic Regression

model_curr <-  data.frame(model_id= 1
                          , model_name= 'softmax - MN logistic reg'
                          , model_alias= 'pred_mod1'
                          , req_pack= 'VGAM'
                          , func = 'vglm'
                          , mod_arg = 'vglm(Formula,family = "multinomial",data=data_in)')

model_meta <- rbind(model_meta, model_curr)

model1 <- vglm(Formula,family = "multinomial",data=data_in_train)


#
# on train
#

probability <- predict(model1,x_train,type="response")

# logistic requires different prediction process
data_in_train$pred_mod1<-apply(probability,1,which.max)



data_in_train$pred_mod1 <- factor(data_in_train$pred_mod1,
                                 levels = fact_levels,
                                 labels = fact_labels)


#
# on test
#
probability <- predict(model1,x_test,type="response")

# logistic requires different prediction process
data_in_test$pred_mod1<-apply(probability,1,which.max)


data_in_test$pred_mod1 <- factor(data_in_test$pred_mod1,
          levels = fact_levels,
          labels = fact_labels)





#2 Linear Discriminant Analysis

model_curr <-  data.frame(model_id= 2
                          , model_name= 'linear discriminant'
                          , model_alias= 'pred_mod2'
                          , req_pack= 'MASS'
                          , func = 'lda'
                          , mod_arg = 'lda(Formula,data=data_in)')


model_meta <- rbind(model_meta, model_curr)

model2<-lda(Formula,data=data_in_train)

data_in_train$pred_mod2<-data.frame(predict(model2,x_train, type="response"))[,tolower(resp_var)]
data_in_test$pred_mod2<-data.frame(predict(model2,x_test, type="response"))[,tolower(resp_var)]

#3) Mixture Discriminant Analysis

model_curr <-  data.frame(model_id= 3
                          , model_name= 'mixture discriminant'
                          , model_alias= 'pred_mod3'
                          , req_pack= 'MASS'
                          , func = 'mda'
                          , mod_arg = 'mda(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)

#Build the model
model3<-mda(Formula,data=data_in_train)

data_in_train$pred_mod3<-predict(model3,x_train)

data_in_test$pred_mod3<-predict(model3,x_test)



####2)Quadratic Discriminant Analysis- error
#model_curr <-  data.frame(model_id= x, model_name= 'quadratic discriminant')
#model_meta <- rbind(model_meta, model_curr)
#model4<-qda(Formula,data=data_in)
#summary(model4)
#data_in$pred_qda<-predict(model4,x)


####3)Regularized Discriminant Analysis
model_curr <-  data.frame(model_id= 4
                          , model_name= 'regularized discriminant'
                          , model_alias= 'pred_mod4'
                          , req_pack= 'MASS'
                          , func = 'rda'
                          , mod_arg = 'rda(Formula,data=data_in,gamma = 0.05,lambda = 0.01)')


model_meta <- rbind(model_meta, model_curr)

model4 <- rda(Formula,data=data_in_train,gamma = 0.05,lambda = 0.01)

data_in_train$pred_mod4<-data.frame(predict(model4,x_train, type="response"))[,tolower(resp_var)]
data_in_test$pred_mod4<-data.frame(predict(model4,x_test, type="response"))[,tolower(resp_var)]

#5 Neural Network
model_curr <-  data.frame(model_id= 5
                          , model_name= 'neural network'
                          , model_alias= 'pred_mod5'
                          , req_pack= 'nnet'
                          , func = 'nnet'
                          , mod_arg = 'nnet(Formula,data=data_in,size = 4,decay = 0.0001,maxit = 500)')

model_meta <- rbind(model_meta, model_curr)


model5 <- nnet(Formula,data=data_in_train,size = 4,decay = 0.0001,maxit = 500)
data_in_train$pred_mod5 <-predict(model5,x_train,type="class")
data_in_test$pred_mod5 <-predict(model5,x_test,type="class")



#6 Flexible Discriminant Analysis
model_curr <-  data.frame(model_id= 6
                          , model_name= 'flexible discriminant'
                          , model_alias= 'pred_mod6'
                          , req_pack= 'MASS'
                          , func = 'fda'
                          , mod_arg = 'fda(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)

model6 <- fda(Formula,data=data_in_train)
data_in_train$pred_mod6 <-predict(model6,x_train,type="class")
data_in_test$pred_mod6 <-predict(model6,x_test,type="class")


#7 Support Vector Machine


model_curr <-  data.frame(model_id= 7
                          , model_name= 'support vector machine'
                          , model_alias= 'pred_mod7'
                          , req_pack= 'kernlab'
                          , func = 'ksvm'
                          , mod_arg = 'ksvm(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)

model7 <- ksvm(Formula,data=data_in_train)
data_in_train$pred_mod7<-predict(model7,x_train,type="response")
data_in_test$pred_mod7<-predict(model7,x_test,type="response")


# 8) k-Nearest Neighbors

model_curr <-  data.frame(model_id= 8
                          , model_name= 'k-nearest neighbors'
                          , model_alias= 'pred_mod8'
                          , req_pack= 'caret'
                          , func = 'knn3'
                          , mod_arg = 'knn3(Formula,data=data_in,k=5)')

model_meta <- rbind(model_meta, model_curr)


model8<-knn3(Formula,data=data_in_train,k=5)
data_in_train$pred_mod8<-predict(model8,x_train,type="class")
data_in_test$pred_mod8<-predict(model8,x_test,type="class")



#9) Naive Bayes

model_curr <-  data.frame(model_id= 9
                          , model_name= 'naive bayes'
                          , model_alias= 'pred_mod9'
                          , req_pack= 'e1071'
                          , func = 'naiveBayes'
                          , mod_arg = 'naiveBayes(Formula,data=data_in,k=5)')

model_meta <- rbind(model_meta, model_curr)


model9 <-naiveBayes(Formula,data=data_in_train,k=5)
data_in_train$pred_mod9 <- predict(model9,x_train)
data_in_test$pred_mod9 <- predict(model9,x_test)


#10) Classification and Regression Trees(CART)

model_curr <-  data.frame(model_id= 10
                          , model_name= 'classification and regression tree'
                          , model_alias= 'pred_mod10'
                          , req_pack= 'rpart'
                          , func = 'rpart'
                          , mod_arg = 'rpart(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)

model10 <- rpart(Formula,data=data_in_train)
data_in_train$pred_mod10<-predict(model10,x_train,type="class")
data_in_test$pred_mod10<-predict(model10,x_test,type="class")


#11) C4.5

model_curr <-  data.frame(model_id= 11
                          , model_name= 'tree C4.5'
                          , model_alias= 'pred_mod11'
                          , req_pack= 'RWeka'
                          , func = 'J48'
                          , mod_arg = 'J48(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)


model11<-J48(Formula,data=data_in_train)
data_in_train$pred_mod11<-predict(model11,x_train)
data_in_test$pred_mod11<-predict(model11,x_test)

#12) PART

model_curr <-  data.frame(model_id= 12
                          , model_name= 'PART'
                          , model_alias= 'pred_mod12'
                          , req_pack= 'RWeka'
                          , func = 'PART'
                          , mod_arg = 'PART(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)


model12<-PART(Formula,data=data_in_train)
data_in_train$pred_mod12<-predict(model12,x_train)
data_in_test$pred_mod12<-predict(model12,x_test)

#13) Bagging CART
model_curr <-  data.frame(model_id= 13
                          , model_name= 'bagging CART'
                          , model_alias= 'pred_mod13'
                          , req_pack= 'ipred'
                          , func = 'bagging'
                          , mod_arg = 'bagging(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)


model13<-bagging(Formula,data=data_in_train)
data_in_train$pred_mod13<-predict(model13,x_train)
data_in_test$pred_mod13<-predict(model13,x_test)


#14) Random Forest

model_curr <-  data.frame(model_id= 14
                          , model_name= 'random forest'
                          , model_alias= 'pred_mod14'
                          , req_pack= 'randomForest'
                          , func = 'randomForest'
                          , mod_arg = 'randomForest(Formula,data=data_in)')

model_meta <- rbind(model_meta, model_curr)

model14<-randomForest(Formula,data=data_in_train)
data_in_train$pred_mod14<-predict(model14,x_train)
data_in_test$pred_mod14<-predict(model14,x_test)


#15) Gradient Boosted Machine

model_curr <-  data.frame(model_id= 15
                          , model_name= 'gradient boosted machine'
                          , model_alias= 'pred_mod15'
                          , req_pack= 'gbm'
                          , func = 'gbm'
                          , mod_arg = 'gbm(Formula,data=data_in,distribution="multinomial")')

model_meta <- rbind(model_meta, model_curr)


model15<-gbm(Formula,data=data_in_train,distribution="multinomial")
probability<-predict(model15,x_train,n.trees=1)
data_in_train$pred_mod15 <- colnames(probability)[apply(probability,1,which.max)]

probability<-predict(model15,x_test,n.trees=1)
data_in_test$pred_mod15 <- colnames(probability)[apply(probability,1,which.max)]


#Boosted C5.0

model_curr <-  data.frame(model_id= 16
                          , model_name= 'boosted C5.0'
                          , model_alias= 'pred_mod16'
                          , req_pack= 'C50'
                          , func = 'C5.0'
                          , mod_arg = 'C5.0(Formula,data=data_in,trials=10)')

model_meta <- rbind(model_meta, model_curr)

#Build the model
model16 <- C5.0(Formula,data=data_in_train,trials=10)
data_in_train$pred_mod16 <- predict(model16,x_train)
data_in_test$pred_mod16 <- predict(model16,x_test)


######################################
# End Models
######################################


######################
# Confusion Matrix - cross tab
######################

## TRAIN
# select prediction variables from data table
myvars <- names(data_in_train) %in% pred_vars
# create list of cross confusion matricies compared to response variable
cm_all <- apply(data_in_train[!myvars], 2, function(x) table(x,eval(parse(text=paste0('data_in_train$',resp_var)))))
cm_all$data_set <- NULL

#Calculate overall accuracy measures per model
oa_1 <- mapply(cm_all, FUN = function(x) tryCatch(confusionMatrix(x)$overall,error = function(err) NA))
oa_1 <-oa_1[!is.na(oa_1)]

#create data frame of model measures
perfom_mod_train <- data.frame(t(data.frame(oa_1)))
perfom_mod_train$model_alias<-rownames(perfom_mod_train)
colnames(perfom_mod_train) <- paste("train", colnames(perfom_mod_train), sep = "_")
colnames(perfom_mod_train)[length(colnames(perfom_mod_train))] <- "model_alias"

## TEST
head(data_in_test)

# select prediction variables from data table
myvars <- names(data_in_test) %in% pred_vars
# create list of cross confusion matricies compared to response variable
cm_all <- apply(data_in_test[!myvars], 2, function(x) table(x,eval(parse(text=paste0('data_in_test$',resp_var)))))
cm_all$data_set <- NULL

#Calculate overall accuracy measures per model
oa_1 <- mapply(cm_all, FUN = function(x) tryCatch(confusionMatrix(x)$overall,error = function(err) NA))
oa_1 <-oa_1[!is.na(oa_1)]

#create data frame of model measures
perfom_mod_test <- data.frame(t(data.frame(oa_1)))
perfom_mod_test$model_alias<-rownames(perfom_mod_test)
colnames(perfom_mod_test) <- paste("test", colnames(perfom_mod_test), sep = "_")
colnames(perfom_mod_test)[length(colnames(perfom_mod_test))] <- "model_alias"

# final data set of model performance
all_model_performance <- merge(merge(model_meta,perfom_mod_train),perfom_mod_test)
all_model_performance$model_name_labels <- paste("(", floor(all_model_performance$test_Kappa*1000)/1000,
                                                 ")- ", all_model_performance$model_name, sep="")

# sort by kappa
all_model_performance_rank <- all_model_performance[order(-all_model_performance$test_Kappa),]

all_model_performance_rank$model_name_labels <- factor(all_model_performance_rank$model_name_labels,
                                                       levels = rev(all_model_performance_rank$model_name_labels[order(all_model_performance_rank$test_Kappa)])
                                                       ,ordered = TRUE)

merge(merge(model_meta,perfom_mod_train, all=TRUE),perfom_mod_test, all=TRUE)

min_kappa <- min(c(all_model_performance$train_Kappa, all_model_performance$test_Kappa))
max_kappa <- max(c(all_model_performance$train_Kappa, all_model_performance$test_Kappa))

###########
# Simple scatterplot
#plot_ly(data = all_model_performance, x = train_Kappa, y = test_Kappa, mode = "markers",
#        color = model_name)

#install.packages("ggplot2")
#library(ggplot2)

p1 <- ggplot(data=all_model_performance_rank, aes(x=train_Kappa))  + 
      xlim(min_kappa*.98, 1.02) + 
      ylim(min_kappa*.98, 1.02) +  
      geom_point(aes(y=test_Kappa, colour=model_name_labels), size=2) +
      geom_abline(intercept = 0, slope = 1, color="black",linetype="dashed", size=0.5)
  

(gg <- ggplotly(p1))


##############################
# STEP 2: simulate robustness of model
#############################



perf_df_train <- data.frame(model_id= integer(0)
                      ,model_name= character(0)
                      ,model_it= integer(0)
                      ,Accuracy= numeric(0)
                      ,Kappa = numeric(0)
                      ,Mean_Sensitivity = numeric(0)
                      ,Mean_Specificity = numeric(0)
                      ,Mean_Pos_Pred_Value = numeric(0)
                      ,Mean_Neg_Pred_Value = numeric(0)
                      ,Mean_Detection_Rate = numeric(0)
                      ,Mean_Balanced_Accuracy = numeric(0)
)

perf_df_test <- perf_df_train



for(i in 1:100){
#  i<-1
  
model_it <- i
train_ratio <- 0.75       #define % of training and test set
trainindex <- sample(1:nrow(data_in),floor(nrow(data_in)*train_ratio))     #Random sample of rows for training set      

data_in_train <- data_in[trainindex, ]   #get training set
data_in_test <- data_in[-trainindex, ]     #get test set                                   



x_train <- data_in_train[,pred_vars]
x_test <- data_in_test[,pred_vars]


#model
model_id <- 7
model_name <- 'support vector machine'
model <- ksvm(Formula,data=data_in_train)
data_in_train$pred_mod<-predict(model,x_train,type="response")
data_in_test$pred_mod<-predict(model,x_test,type="response")


#TRAIN performance

data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
names(data_tmp) <- c("obs", "pred")
tmp <- cbind(model_id , model_name, model_it, data.frame(
  t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
))
perf_df_train <- rbind(perf_df_train,tmp)


#TEST performance

data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
names(data_tmp) <- c("obs", "pred")
tmp <- cbind(model_id , model_name, model_it, data.frame(
  t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
))
perf_df_test <- rbind(perf_df_test,tmp)

#model
model_id <- 8
model_name <- 'k-nearest neighbors'
model <- knn3(Formula,data=data_in_train,k=5)
data_in_train$pred_mod <-predict(model,x_train,type="class")
data_in_test$pred_mod <-predict(model,x_test,type="class")


#TRAIN performance

data_tmp <- data_in_train[,c(resp_var, "pred_mod")]
names(data_tmp) <- c("obs", "pred")
tmp <- cbind(model_id , model_name, model_it, data.frame(
  t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_train$',resp_var))))))
))
perf_df_train <- rbind(perf_df_train,tmp)


#TEST performance

data_tmp <- data_in_test[,c(resp_var, "pred_mod")]
names(data_tmp) <- c("obs", "pred")
tmp <- cbind(model_id , model_name, model_it, data.frame(
               t(multiClassSummary(data_tmp, lev = levels(eval(parse(text=paste0('data_in_test$',resp_var))))))
               ))
perf_df_test <- rbind(perf_df_test,tmp)


}




## Visualize the differences

# histogram comparisons
d <- melt(perf_df_test[,-c(1,3)])
ggplot(d,aes(x = value, fill=model_name)) + 
  facet_wrap(~variable,scales = "free_x") +  
  geom_histogram()


#boxplot(perf_df[,-c(1:3)])

# single variable box plots - can extend to all models at once
d <- melt(perf_df_test[,c("model_name", "Accuracy")])

ggplot(d, aes(x=model_name, y=value)) + geom_boxplot()