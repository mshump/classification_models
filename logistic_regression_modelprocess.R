
##########################
# Install Packages (1 time)
##########################
#install.packages("aod")
#install.packages("ggplot2")
#install.packages("Rcpp")
#install.packages("plyr")
#install.packages("reshape2")
#install.packages("rpart")

#install.packages("CHAID", repos="http://R-Forge.R-project.org") ## This does not work for me

##########################
# Load packages 
##########################
library(aod)
library(ggplot2)
library(Rcpp)
library(plyr)
library(reshape2)
library(rpart)
library(CHAID)


##########################
# Load .csv file
##########################
mydata <- read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")
## view and summarize the data
str(mydata)
head(mydata)
tail(mydata)




#############################
# change data types if needed
############################
mydata$rank <- factor(mydata$rank)                  
mydata_test1 <- mydata[mydata$rank == 1, ]
mydata_test2 <- mydata[mydata$rank == 2, ]

mydata_test <- mydata[mydata$rank %in% c(1,2), ]

plot(ctree(admit  ~ gre,  mydata), type = "simple")

######################################
# explore some summaries of data
######################################
summary(mydata)

sapply(mydata, sd)

xtabs(~ admit + rank, data = mydata)

ddply(mydata,~admit,summarise,mean=mean(gre),sd=sd(gre))

aggregate(mydata$gre, list(mydata$admit), mean)


#prevalence rate of positive outcome (outcome = 1)
prev_rate_all <- prop.table(table(mydata$admit))[[2]] 


######################################
# Some basic visualizations
######################################
#
str(mydata)

# some single viz
hist(mydata$gre)
cts <-table(as.factor(mydata$admit))
barplot(cts)
cts <-table(mydata$rank)
barplot(cts)

# get some histos
resp <- 1 #column response
d <- melt(mydata[,-resp], id="rank")
ggplot(d,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()



######################################
# Split data to test and training sets
######################################
# within population , not based on future time period
# set.seed(3456)



train_ratio <- 0.75       #define % of training and test set
trainindex <- sample(1:nrow(mydata),floor(nrow(mydata)*train_ratio))     #Random sample of rows for training set      

mydata_train <- mydata[trainindex, ]   #get training set
mydata_train$data_set <- 'train' 
mydata_test <- mydata[-trainindex, ]     #get test set                                   
mydata_test$data_set <- 'test'

# Hows are response distribution between data sets?
# Want to be similar
prev_train <- table(mydata_train$admit)
prop.table(table(mydata_train$admit)) 
prev_test <- table(mydata_test$admit)
prop.table(table(mydata_test$admit)) 

# quick test on difference in response prevalance rates
chi2 <- chisq.test( rbind(prev_train,prev_test)) 
chi2$p.value  

######################################
# Investigate multicolinearity
######################################

# Multivariate , bi-variate Scatterplot Matrix
pairs(~gre+gpa+rank,data=mydata_train, 
      main="Simple Scatterplot Matrix")


# single bi-variate Scatterplot
plot(mydata_train$gre, mydata_train$gpa, main="Scatterplot Example", 
     xlab="gre", ylab="gpa", pch=19)


# Add linear fitetd line
abline(lm(mydata_train$gpa~ mydata_train$gre), col="red") # regression line (y~x) 
lines(lowess(mydata_train$gre, mydata_train$gpa), col="blue") # lowess line (x,y)

# Assess R squared (relationship between numeric variables)
# if is close to 1, then risk to model
lm1 <- lm(data=mydata_train,formula = gre ~ gpa)
summary(lm1)
lm1_r2 <- summary(lm1)$r.squared
lm1_r2

######################################
# Develop bivariate assesments with response
######################################
# dependent is discrete / binomial (churn)
# if independent variable 

tst <- glm(admit ~ gre , data = mydata_train, family = "binomial")
tst <- glm(admit ~ gpa , data = mydata_train, family = "binomial")
tst <- glm(admit ~ rank , data = mydata_train, family = "binomial")

summary(tst)


mydata$admit_fact <- as.factor(mydata$admit)  
mydata$gre_fact <- as.factor(mydata$gre) 

ctrl <- chaid_control(minsplit = 1, minbucket = 20, minprob = 0)

chaid_run <- chaid(admit_fact ~ gre_fact, data = mydata, control = ctrl)
?chaid
str(chaid_run)
print(chaid_run)
summary(chaid_run)

## Plot the decision tree

plot(chaid_run)

## get rule path
partykit:::.list.rules.party(chaid_run)

  
  format_rules <- function(object, ...) {
    ft <- fitted(object)
    ns <- tapply(ft[[2]], ft[[1]], length)
    pr <- tapply(ft[[2]], ft[[1]], function(y)
      min(prop.table(table(y))))
    lb <- tapply(ft[[2]], ft[[1]], function(y)
      names(sort(table(y), decreasing = TRUE))[1])
    rl <- partykit:::.list.rules.party(object)
    paste0(rl, ": ", lb, " (n = ", ns, ", ", round(100 * pr, 2), "%)")
  }
writeLines(format_rules(chaid_run))

#####################

## read data with factor response
d <- read.table("text.txt", header = TRUE)
d$Rating <- factor(d$Rating)

## ctree
library("partykit")
ct <- ctree(admit ~ gre, data = mydata)
writeLines(format_rules(ct))
plot(ct)

## rpart
library("rpart")
rp <- rpart(admit ~ gre, data = mydata, control = list(cp = 0.02))
writeLines(format_rules(rp))
plot(as.party(rp))

## evtree
install.packages("evtree")
library("evtree")
set.seed(1)
ev <- evtree(admit ~ gre, data = mydata, maxdepth = 5)
writeLines(format_rules(ev))
plot(ev)

#######################

# Boxplot 
boxplot(gre~admit,data=mydata_train)


# Boxplot 
cts <-table(as.factor(mydata$admit), as.factor(mydata$rank))
barplot(cts, beside=TRUE, legend = rownames(cts))

cts <-table(as.factor(mydata$rank), as.factor(mydata$admit))
barplot(cts, beside=TRUE, legend = rownames(cts))


######################################
# Develop logistic model
######################################
# talk with matt -

mylogit_train <- glm(admit ~ gre + gpa + rank, data = mydata_train, family = "binomial")

summary(mylogit_train)


######################################
# Apply predictions
######################################
mylogit_train_prob <- predict(mylogit_train, mydata_train, type="response")

mylogit_test_prob <- predict(mylogit_train, mydata_test, type="response")


######################################
# Asses prediction performance
######################################

## with training set -
mydata_train_asses <- mydata_train

mydata_train_asses$pred_prob <- predict(mylogit_train, mydata_train, type="response")

mydata_test_asses <- mydata_test

mydata_test_asses$pred_prob <- predict(mylogit_train, mydata_test, type="response")

mydata_asses <- rbind(mydata_train_asses,mydata_test_asses)

mydata_asses$pred <- ifelse(mydata_asses$pred_prob >0.5,1,0)

mydata_asses$pred_correct <-  ifelse(mydata_asses$admit == mydata_asses$pred,1,0) 


cut_points <- seq(from=0, to=1, by=0.1)

mydata_asses$pred_cut <- (cut(mydata_asses$pred_prob, breaks=cut_points, labels= FALSE ) -1)/ (length(cut_points)-1)



############################################
#Some quick model performance summary stats#
############################################

sum(as.numeric(mydata_asses[mydata_asses$data_set == 'train', 'pred_correct']) ) / length(as.numeric(mydata_asses[mydata_asses$data_set == 'train', 'pred_correct']))
sum(as.numeric(mydata_asses[mydata_asses$data_set == 'test', 'pred_correct']) )  / length(as.numeric(mydata_asses[mydata_asses$data_set == 'test', 'pred_correct']))



model_eval <- aggregate.data.frame(mydata_asses[, c('admit','pred_correct')]
                     , list(mydata_asses[,'pred_cut'],mydata_asses[,'data_set'])
                     , function(x) c(crt=sum(x),n=length(x) ,rate=sum(x)/length(x)))


#df.eval_line <- as.data.frame(cbind(Group.1=as.numeric(cut_points)
#                    ,Group.2 =rep('eval_line',length(cut_points))
#                    ,admit.rate =as.numeric(cut_points)
#                    ,pred_correct.rate =rep(prev_rate_all,length(cut_points))))


                
write.table(model_eval, file = "C:/Users/Matt/Documents/Projects/EIG/HG Churn/outputs/model_eval.csv", sep = ",")



##################
#Model parameters#
##################


d_obs <- mydata_asses[1,]

predict(mylogit_train, mydata_asses[1,], type="response")

str(mylogit_train)

coefs <- coef(mylogit_train)

model_math <- (coefs["(Intercept)"] 
                  + d_obs$gre*coefs["gre"] 
                  + d_obs$gpa*coefs["gpa"] 
                  + (d_obs$rank==2)*coefs["rank2"] 
                  + (d_obs$rank==3)*coefs["rank3"]
                  + (d_obs$rank==4)*+ coefs["rank4"]               
               )[[1]]
    
model_prob <-  exp(model_math) / (1+exp(model_math))





## odds ratios only
exp(coef(mylogit_train))

summary(mylogit_train)

## odds ratios and 95% CI
exp(cbind(OR = coef(mylogit_train), confint(mylogit_train)))


##################
#model evaluations#
##################


## CIs using profiled log-likelihood
confint(mylogit_train)


## CIs using standard errors
confint.default(mylogit_train)


wald.test(b = coef(mylogit_train), Sigma = vcov(mylogit_train), Terms = 4:6)


l <- cbind(0,0,0,1,-1,0)
wald.test(b = coef(mylogit_train), Sigma = vcov(mylogit_train), L = l)



## 
# add in residual and outlier analysis
##

