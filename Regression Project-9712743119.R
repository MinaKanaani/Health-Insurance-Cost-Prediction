#Rising health care costs are a major economic and public health issue worldwide : According to the World Health Organization,
#health care accounted for 7.9% of Europe's gross domestic product (GDP) in 2015. In Switzerland, the health care sector contributes
#substantially to the national GDP, and has increased from 10.7 to 12.1% between 2010 and 2015. Moreover, because health care
#utilization costs may serve as a surrogate for an individual's health status, understanding which factors contribute to 
#increases in health expenditures may provide insight into risk factors and potential starting points for preventive measures.

#Several studies have addressed the prediction of health care costs, approaching the issue as either a regression problem 
#or a classification problem (classifying costs into predefined "buckets"). Previous studies also examined a broad variety 
#of features. The most commonly used features include different sets of demographic features, health care utilization parameters
#(e.g. hospitalization or outpatient visits), drug codes, diagnosis codes, procedure codes, various chronic disease scores and 
#cost features.

#In this study, we aim to predict the health charges of Bangladeshi citizens based on some important features such as their age,
#sex, BMI, number of children they have, etc. We approached the problem as a binary classification task, correlating each features 
#with charges in order to capture different patterns of the data. Based on their characteristics, 
#we built two models: Linear Regression and Logistic Regression Model. Finally, we tried out some tests to examine 
#the prediction which showed us good results.

#https://medium.datadriveninvestor.com/10-machine-learning-projects-on-classification-with-python-9261add2e8a7


library(DiagrammeR)

DiagrammeR::grViz("digraph {

graph [layout = dot, rankdir = TB, label = 'Project Flowchart',labelloc= ta]
node [shape = rectangle, style = filled, fillcolor = dodgerblue]

'Calling libraries' -> 'EDA analysis' -> 'Processing data' -> 'Linear model'
'Processing data' -> 'KNN model'
'Processing data' -> 'Regression Tree model'
'Linear model'-> Visualization
'KNN model'-> Visualization
'Regression Tree model'-> Visualization
Visualization->'RMSE candid model'
}")
#needed libraries
library(readxl)
library(tibble)   # data frame printing
library(dplyr)      # data manipulation
library(caret)     # fitting knn
library(rpart)    # fitting trees
library(rpart.plot) # plotting trees
library(knitr)
library(Hmisc)
library(cowplot)
#library(cowplot)
#library(WVPlots)
#EDA needed library
library(ggplot2)


#reading the file
insu_data=read_excel("insurance.xlsx")
insu_data
attach(insu_data)
#-----------------------------------------------------------------------------------------------------
#EDA visualization analysis
describe(insu_data)
sum(is.na(insu_data))
#first plot
par(mfrow=c(1,2))
plot1_1=plot(charges~age,col='light green',pch=20,cex=1, main="Correlation : Charges and Age")+
  geom_jitter()
grid()
plot1_2=plot(charges~bmi,col='purple',pch=20,cex=1, main="Correlation : Charges and bmi")+
  geom_jitter()
grid()
#as we can see , as the age and bmi increase, the charges increase too. so there is a trend up here.
#second plot
plot2_1=ggplot(insu_data, aes(sex, charges)) +
  geom_jitter(aes(color = sex), alpha = 0.7) +
  theme_light()
plot2_2=ggplot(insu_data, aes(children, charges)) +
  geom_jitter(aes(color = children), alpha = 0.7) +
  theme_light()

plot2_3=plot_grid(plot2_1,plot2_2)
title2=ggdraw()+draw_label("Correlation between Charges and children/Sex")
plot2=plot_grid(title2,plot2_3,ncol=1,rel_heights=c(0.1, 1))
#as illustrated, gender does not apparently have any effect on the charges, 
#with number of children, we can see a decreasing trend, the more children, the less the charges which is odd 
#and doesn't seem logical.

#third plot
plot3_1=ggplot(insu_data,aes(region,charges))+
  geom_jitter(aes(col=region),alpha=0.7)+
  theme_light()

plot3_2=ggplot(insu_data,aes(smoker,charges))+
  geom_jitter(aes(col=smoker),alpha=0.7)+
  theme_light()
plot3_3=plot_grid(plot3_1,plot3_2)
title3=ggdraw()+draw_label("Correlation between Charges and Region/Smoker")
plot3=plot_grid(title3,plot3_3,ncol=1,rel_heights=c(0.1, 1))
par(mfrow=c(1,1))
# as it is illustrated, no obvious connection can be seen between region and charges, also,as we expected,
#charges for smokers are higher than non_smokers.
#--------------------------------------------------------------------------------------------
#processing data
#AS we saw in above visualizations, we have Two not significant variables, "sex" and "region".
#so I decided to remove them from the main data.
insu_data=subset(insu_data,select = -c(sex,region))
#split Train and test

set.seed(42)
insu_trn_idx = sample(nrow(insu_data), size = 0.8 * nrow(insu_data))
insu_trn = insu_data[insu_trn_idx, ]
insu_tst = insu_data[-insu_trn_idx, ]

#split estimation and validation 
insu_est_idx = sample(nrow(insu_trn), size = 0.8 * nrow(insu_trn))
insu_est = insu_trn[insu_est_idx, ]
insu_val = insu_trn[-insu_est_idx, ]

#check data
head(insu_trn, n = 10)

skimr::skim(insu_trn)
str(insu_trn)
#View(insu_trn)

GGally::ggpairs(insu_trn, progress = FALSE)
#----------------------------------------------------------------------------------------------------------
#linear_models

insu_mod_list=list(
  insu_mod_1=lm(charges~1,data=insu_est),
  insu_mod_2=lm(charges~age+bmi,data=insu_est),
  insu_mod_3=lm(charges~age+bmi+smoker,data=insu_est),
  insu_mod_4=lm(charges~.,data=insu_est),
  insu_mod_5=step(lm(charges ~ . ^ 2, data = insu_est), trace = FALSE),
  insu_mod_6=lm(charges~poly(bmi,degree=2),data=insu_est),
  insu_mod_7=lm(charges~poly(age,degree=2),data=insu_est)
  
)
attach(insu_mod_list)
#RMSE Function
calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
#select candid model
par(mfrow=c(1,1))
insu_mod_val_pred = lapply(insu_mod_list, predict, insu_val )
val_RMSE=sapply(insu_mod_val_pred, calc_rmse, actual = insu_val$charges) 
barplot(val_RMSE,col='pink',beside=T,legend.text = T)
#as we  compare the validation RMSE, we can see the 5th model consist of all variables with their binary intercations with 
#each other is the candid model with least RMSE.


#visualization
plot(
  x = insu_val$charges,
  y = predict(insu_mod_5, insu_val),
  pch = 20, col = "green",
  main = "Credit: Predicted vs Actual, Test Data",
  xlab = "Actual",
  ylab = "Predicted"
)
abline(a=0,b=1, lwd = 2)
grid()
#------------
#---------------------------------------------------------------------------------------------------------------
#KNN
# Before we start with fitting candidate models of KNN we have to use Scale function to normalize the quanitifiable features such
#as age and bmi.
#age
insu_est$age.s=scale(insu_est$age)
age.center = attr(insu_est$age.s,"scaled:center")
age.scale = attr(insu_est$age.s,"scaled:scale")
#bmi
insu_est$bmi.s=scale(insu_est$bmi)
bmi.center = attr(insu_est$bmi.s,"scaled:center")
bmi.scale = attr(insu_est$bmi.s,"scaled:scale")

# normalization of validation data
#age
insu_val$age.s = scale(insu_val$age, center = age.center, scale = age.scale)
#bmi
insu_val$bmi.s = scale(insu_val$bmi, center = bmi.center, scale = bmi.scale)

#tunning K 
#now after we normalized our features, I created a function to generate K ranging from 1 to 100 and after comparing the RMSEs,
#we fit the final model.

knn_mod_list=list()
creat_knn_normal=function(i){
  knnreg(charges ~ age.s +smoker + bmi.s+children, data = insu_est, k =i)
}
for(i in 1:100){
  knn_mod_list[i]=list(creat_knn_normal(i))
}
length(knn_mod_list)

#fitting candid models
knn_val_pred=lapply(knn_mod_list,predict,insu_val)
# model selection
knn_val_rmse=sapply(knn_val_pred, calc_rmse, insu_val$charges)
opt_mode=which.min(knn_val_rmse)
insu_KNN_06=knn_mod_list[[opt_mode]]

#the model with k=6 is the best one with lowest RMSE
knn_table_normal=data.frame(knn_val_rmse)
knn_table_normal$K=c(1:100)
knn_table_normal=knn_table_normal%>%select(-knn_val_rmse,everything())
knn_table_normal


#now for comparison and better analysis, we fit the models without normalization first to see the difference.
# not normal
knn_mod_list_not=list()
creat_knn_not=function(i){
  knnreg(charges ~ age +smoker + bmi+children, data = insu_est, k =i)
}
for(i in 1:100){
  knn_mod_list_not[i]=list(creat_knn_not(i))
}
length(knn_mod_list_not)

#fitting candid models
knn_val_pred_not=lapply(knn_mod_list_not,predict,insu_val)
# model selection
knn_val_rmse_not=sapply(knn_val_pred_not, calc_rmse, insu_val$charges)
opt_mode_not=which.min(knn_val_rmse_not)
insu_KNN_02=knn_mod_list_not[[opt_mode_not]]

#the model with k=2 is the best one with lowest RMSE
knn_table_not=data.frame(knn_val_rmse_not)
knn_table_not$K=c(1:100)
knn_table_not=knn_table_not%>%select(-knn_val_rmse_not,everything())
knn_table_not

#visualization
par(mfrow=c(1,1))
plot(knn_table_normal,xlab="K",ylab="RMSE",main="RMSE  normal", pch = 20,cex=1,col = "dodgerblue",ylim=c(6000,13000))
points(knn_table_not,xlab="K",ylab="RMSE",main="RMSE not normal", pch = 20,cex=1,col = "red")
legend(60,7000,legend = c('normalized data','nor normalized data'),col=c('dodgerblue','red'),cex=1,lty=1)
#as it is illustrated in the plot above, we can see that not normalized RMSEs are higher than normalized one in any K so we can
#conclude that normalizing quantifiable features have signifiacant effect on RMSE and therefore gradual candid model.



#visualization
plot(
  x = insu_val$charges,
  y = predict(knn_mod_list[[6]], insu_val),
  pch = 20, col = "blue",
  main = "Credit: Predicted vs Actual, Test Data",
  xlab = "Actual",
  ylab = "Predicted"
)
abline(a = 0, b = 1, lwd = 2)
grid()
#---------------------------------------------------------------------------------------------------------------------------
#REGRESSION TREE
#with regression tree ,I create a list with 10 models, 5 consist of minsplit =5 and cp ranging from 0 to 1.

tree_mod_list = list(
  insu_tree_1 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.000,minsplit=5),
  insu_tree_2 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.001,minsplit=5),
  insu_tree_3 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.01,minsplit=5),
  insu_tree_4 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.1,minsplit=5),
  insu_tree_5 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 1,minsplit=5),
  insu_tree_6 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.000,minsplit=20),
  insu_tree_7 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.001,minsplit=20),
  insu_tree_8 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.01,minsplit=20),
  insu_tree_9 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 0.1,minsplit=20),
  insu_tree_10 = rpart(charges ~ age +smoker + bmi+children, data = insu_est, cp = 1,minsplit=20)
)

attach(tree_mod_list)

tree_val_predict=lapply(tree_mod_list,predict,insu_val)
tree_val_rmse=sapply(tree_val_predict,calc_rmse,insu_val$charges)
opt_tree_mod=tree_mod_list[which.min(tree_val_rmse)]
#the tree model 7 with cp = 0.001, minsplit=20 is the candid model with lowest RMSE.
tree_table=data.frame(tree_val_rmse)
tree_table$minsplit=c(rep(20,5),rep(5,5))
tree_table$cp=c(rep(c(0,0.001,0.01,0.1,1),1))
kable(tree_table)

insu_tree = rpart(charges ~ age +smoker + bmi+children, data = insu_est ,cp=0.001,minsplit=20)
insu_tree
rpart.plot(insu_tree)

#analysis: as we can see in the plot, most frequent cut offs are based on aget so we can say it's our most important
#variables, following there are several splits with bmi variable which indicates that these two quantity variables are important.
#there is one first split on smoker which as we expect should have serious effect on insurance costs and  splits on children
#which we can conclude that all 4 features are effective.
#for example, from the first split, we check if the person is a smoker or not,if they are smokers, we
# go to the left side of the chart, now if the person age is lower that 43 ,we go to the lest part and we
#check the number of children they have, if it is more than 1, we go to right side and check again if the number of children is
#more than 2, this part we assume that the number is lower than 2, so we go to left ,there is another split on age to check whether
#they are under 20 or not , if they are older than 20 years old , we can predict that the insurance charge is 6191 and 
#only 10% of all the reported data have the same conditions.




plot(
  x = insu_val$charges,
  y = predict(insu_tree_7, insu_val),
  pch = 20, col = "purple",
  main = "Credit: Predicted vs Actual, Test Data",
  xlab = "Actual",
  ylab = "Predicted"
)
abline(a = 0, b = 1, lwd = 2)
grid()
#---------------------------------------------------------------------------------------------------------------------------
#visulaziation of 3 models
#in this part , we put together the tree candid models from linear, knn, regression tree method and then by calculating RMSE 
#we choos our final optimal model.

par(mfrow = c(1, 3))
#linear
plot(
  x = insu_val$charges,
  y = predict(insu_mod_5, insu_val),
  pch = 20, col = "darkgreen",
  main = "Credit: Predicted vs Actual, Test Data",
  xlab = "Actual",
  ylab = "Predicted"
)
abline(a = 0, b = 1, lwd = 2)
grid()
#KNN
plot(
  x = insu_val$charges,
  y = predict(knn_mod_list[[6]], insu_val),
  pch = 20, col = "blue",
  main = "Credit: Predicted vs Actual, Test Data",
  xlab = "Actual",
  ylab = "Predicted"
)
abline(a = 0, b = 1, lwd = 2)
grid()
#TREE
plot(
  x = insu_val$charges,
  y = predict(insu_tree_7, insu_val),
  pch = 20, col = "orange",
  main = "Credit: Predicted vs Actual, Test Data",
  xlab = "Actual",
  ylab = "Predicted"
)
abline(a = 0, b = 1, lwd = 2)
grid()

#picking up the best model out of 3 options.
optimal_models=list(insu_mod_5,knn_mod_list[[6]],insu_tree_7)
optimal_val_pred=lapply(optimal_models,predict,insu_val)
RMSE_all=sapply(optimal_val_pred,calc_rmse,actual=insu_val$charges)
which.min(RMSE_all)
#The best model out of three is the linear model "insu_mod_5" which consist of all the features with their binary 
#interactions. as we can see the two models of regression tree and linear have close RMSE and both seem to be good models.
#At the end we fit the candid model on the test data to calculate the RMSE.
final_model=step(lm(charges ~ . ^ 2, data = insu_trn), trace = FALSE)
predict_insu=predict(final_model,insu_tst)
calc_rmse(insu_tst$charges,predict_insu)
# the RSME is 586.464
par(mfrow=c(1,1))
