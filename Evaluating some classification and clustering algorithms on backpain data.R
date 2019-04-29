rm(list=ls())
dev.off()

# Loading the data
load("backpain.RData")

# install.packages("rpart")
library(rpart)

# install.packages("adabag")
library(adabag)

# install.packages("randomForest")
library(randomForest)

# Libraries for creating ROC curve and for choosing best threshold values 
library(pROC)
library(scales)

# install.packages('rpart')
library(rpart)

# Missing data contolling
apply(is.na(dat),2,sum)
sum(is.na(dat))

head(dat)
summary(dat)

# plots to understand the data
plot(dat$Age, dat$PainDiagnosis, col=c("red","blue")[dat$PainDiagnosis], main = "Age and Diagnosis", ylab = "Pain Diagnosis", xlab = "Age")
plot(dat$Gender, dat$PainDiagnosis, main = "Gender and Diagnosis", xlab = "Gender", ylab = "Pain Diagnosis")
histogram(dat$PainLocation)
histogram(dat$DurationCurrent)

#****
# Split data into three sets
N = nrow(dat)
train_ind = sort(sample(1:N,size=floor(N*0.70)))
nottestind = setdiff(1:N,train_ind)
valid_ind = sort(sample(nottestind,size=length(nottestind)/2))
test_ind = sort(setdiff(nottestind,valid_ind))
#****


######### STAT40590 CLASSIFICATION PART ###########
# ********** RANDOM FOREST **************
# Fitting randomForest
classifier = randomForest(x = dat[train_ind,-1],
                          y = dat[train_ind,]$PainDiagnosis,
                          ntree = 50)
# Predicting the test set results
y_pred_RF = predict(classifier, newdata = dat[valid_ind,-1])
# Creating prediction results with type prob for ROC curve
y_pred_ROC = predict(classifier, type = "prob", newdata = dat[valid_ind,-1])

# Making the Confusion Matrix and calculating accuracy rate
cm_rf = table(dat[valid_ind,]$PainDiagnosis, y_pred_RF)
cm_rf
acc_rf = sum(diag(cm_rf))/sum(cm_rf)

# Creating ROC curve for random forest
par(pty = "s")
analysis_rf = roc(dat[valid_ind,]$PainDiagnosis, y_pred_ROC[1:nrow(y_pred_ROC)], plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", main= "RF ROC Curve", col="#377eb8", lwd=4, print.auc=TRUE)
par(pty = "m")

# Finding optimal t that minimizes error
e <- cbind(analysis_rf$thresholds,analysis_rf$sensitivities+analysis_rf$specificities)
opt_t_rf <- subset(e,e[,2]==max(e[,2]))[,1]

# Printing optimal threshold value
cat("Optimal threshold value: ", percent(opt_t_rf))

library(ROCR)
#********Showing optimal threshold on ROC curve********
predobj<-prediction(y_pred_ROC[,1],dat[valid_ind,]$PainDiagnosis)
perf <- performance(predobj,"tpr","fpr")

# Plot the ROC curve with optimal threshold value highlighted on the ROC curve
plot(perf, colorize=TRUE,print.cutoffs.at=opt_t_rf, text.adj=c(-0.2,1.7))




# ********** CLASSIFICATION (DECISION) TREE **************
# Fitting classification tree using rpart library
classifier = rpart(formula = PainDiagnosis ~ ., data = dat[train_ind,])
library("partykit")
plot(as.party(classifier))

# Predicting the test set results
y_pred_CF = predict(classifier, newdata = dat[valid_ind,-1], type = 'prob')
pred_corrected = ifelse(y_pred_CF[,1] < 0.5, 1, 0)

# # Creating prediction results with type prob for ROC curve
# y_pred_ROC = predict(classifier, type = "prob", newdata = dat[valid_ind,-1])

# Making the Confusion Matrix and calculating accuracy rate
cm_ct = table(dat[valid_ind,]$PainDiagnosis, pred_corrected)
cm_ct
acc_ct = sum(diag(cm_ct))/sum(cm_ct)

# Creating ROC curve for random forest
par(pty = "s")
analysis_ct = roc(dat[valid_ind,]$PainDiagnosis, pred_corrected, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", main= "Classification Tree ROC Curve", col="#377eb8", lwd=4, print.auc=TRUE)
par(pty = "m")

# Finding optimal t that minimizes error
e <- cbind(analysis_ct$thresholds,analysis_ct$sensitivities+analysis_ct$specificities)
opt_t_ct <- subset(e,e[,2]==max(e[,2]))[,1]

# Printing optimal threshold value
cat("Optimal threshold value: ", percent(opt_t_ct))




# ********** SVM **************
# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = PainDiagnosis ~ .,
                 data = dat[train_ind,],
                 type = 'C-classification',
                 kernel = 'linear', probability = TRUE)

# Predicting the test set results
y_pred_SVM = predict(classifier,type="prob", newdata = dat[valid_ind,-1], probability = TRUE)

# Making the Confusion Matrix and calculating accuracy rate
cm_svm = table(dat[valid_ind,]$PainDiagnosis, y_pred_SVM)
cm_svm
acc_svm = sum(diag(cm_svm))/sum(cm_svm)

# Creating ROC curve for SVM
par(pty = "s")
analysis_svm = roc(dat[valid_ind,]$PainDiagnosis, attr(y_pred_SVM,"probabilities")[,1], plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", main= "SVM ROC Curve", col="#377eb8", lwd=4, print.auc=TRUE)
par(pty = "m")

# Finding optimal t that minimizes error
e <- cbind(analysis_svm$thresholds,analysis_svm$sensitivities+analysis_svm$specificities)
opt_t_svm <- subset(e,e[,2]==max(e[,2]))[,1]

# Printing optimal threshold value
cat("Optimal threshold value: ", percent(opt_t_svm))



# *********** BAGGING *************
# Fitting bagging
classifier = bagging(PainDiagnosis~.,data=dat[train_ind,])

# Predicting the test set results
y_pred_bag = predict(classifier, newdata = dat[valid_ind,-1])
pred_corrected = ifelse(y_pred_bag$prob[,1] < 0.5, 1, 0)
# # Creating prediction results with type prob for ROC curve
# y_pred_ROC = predict(classifier, type = "prob", newdata = dat[valid_ind,-1])

# Making the Confusion Matrix and calculating accuracy rate
cm_bag = table(dat[valid_ind,]$PainDiagnosis, pred_corrected)
cm_bag
acc_bag = sum(diag(cm_bag))/sum(cm_bag)

# Creating ROC curve for bagging
par(pty = "s")
analysis_bag = roc(dat[valid_ind,]$PainDiagnosis, y_pred_bag$prob[,1], plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", main= "Bagging ROC Curve", col="#377eb8", lwd=4, print.auc=TRUE)
par(pty = "m")

# Finding optimal t that minimizes error
e <- cbind(analysis_bag$thresholds,analysis_bag$sensitivities+analysis_bag$specificities)
opt_t_bag <- subset(e,e[,2]==max(e[,2]))[,1]

# Printing optimal threshold value
cat("Optimal threshold value: ", percent(opt_t_bag))





# ********** LOGISTIC REGRESSION **************
head(dat)
summary(dat)
dat_log = dat

# Ensure that any categorical variables are coded as factor
dat_log$PainDiagnosis<-as.factor(dat$PainDiagnosis)
dat_log$Gender<-as.factor(dat$Gender)
dat_log$Criterion2<-as.factor(dat$Criterion2)
dat_log$Criterion4<-as.factor(dat$Criterion4)
dat_log$Criterion6<-as.factor(dat$Criterion6)
dat_log$Criterion7<-as.factor(dat$Criterion7)
dat_log$Criterion8<-as.factor(dat$Criterion8)
dat_log$Criterion9<-as.factor(dat$Criterion9)
dat_log$Criterion10<-as.factor(dat$Criterion10)
dat_log$Criterion13<-as.factor(dat$Criterion13)
dat_log$Criterion19<-as.factor(dat$Criterion19)
dat_log$Criterion20<-as.factor(dat$Criterion20)
dat_log$Criterion26<-as.factor(dat$Criterion26)
dat_log$Criterion28<-as.factor(dat$Criterion28)
dat_log$Criterion32<-as.factor(dat$Criterion32)
dat_log$Criterion33<-as.factor(dat$Criterion33)
dat_log$Criterion36<-as.factor(dat$Criterion36)
dat_log$DurationCurrent<-as.factor(dat$DurationCurrent)
dat_log$PainLocation<-as.factor(dat$PainLocation)


# Fit the logistic regression model
fit_logreg = glm(PainDiagnosis ~ ., data=dat_log, family="binomial")
# Look at the output
summary(fit_logreg)

# Fitting logistic regression with chosen significant variables
fit_logreg2 = glm(PainDiagnosis ~ PainLocation + Criterion4 + Criterion6 + Criterion8 + Criterion9 + Criterion20, data=dat_log[train_ind,], family="binomial")
summary(fit_logreg2)

# Look at predicted probabilities
pred_logReg = predict(fit_logreg2, newdata = dat_log[valid_ind,-1], type="response")
pred_corrected = ifelse(pred_logReg > 0.5, 1, 0)

# Making the Confusion Matrix and calculating accuracy rate
cm_logr = table(dat_log[valid_ind,]$PainDiagnosis, pred_corrected)
cm_logr
acc_logr = sum(diag(cm_logr))/sum(cm_logr)


# Creating ROC curve for LogReg
par(pty = "s")
analysis_logr = roc(dat[valid_ind,]$PainDiagnosis, pred_corrected, plot=TRUE, legacy.axes=TRUE, percent=TRUE, xlab="False Positive Percentage", ylab="True Postive Percentage", main= "LogReg ROC Curve", col="#377eb8", lwd=4, print.auc=TRUE)
par(pty = "m")

# Finding optimal t that minimizes error
e <- cbind(analysis_logr$thresholds,analysis_logr$sensitivities+analysis_logr$specificities)
opt_t_logr <- subset(e,e[,2]==max(e[,2]))[,1]

# Printing optimal threshold value
cat("Optimal threshold value: ", percent(opt_t_logr))

# Odds ratio
beta<-coef(fit_logreg2)
odd_ratios = exp(beta)

summ = summary(fit_logreg2)
betaLB = summ$coef[,1]-qt(0.975, summ$df.residual)*summ$coef[,2]
betaUB = summ$coef[,1]+qt(0.975, summ$df.residual)*summ$coef[,2]
BETA = cbind(betaLB,beta,betaUB)
BETA

# Compute odds & confidence limits for odds
exp(BETA)

# Putting important results through the application of classification methods into vectors
# Accuracy rates, optimal threshold values and AUC values, respectively
acc_vec = c(acc_rf, acc_ct, acc_bag, acc_svm, acc_logr)
opt_t_vec = c(opt_t_rf, opt_t_ct, opt_t_bag, opt_t_svm, opt_t_logr)
auc_vec = c(analysis_rf$auc, analysis_ct$auc, analysis_bag$auc, analysis_svm$auc, analysis_logr$auc)

cm_rf
cm_ct
cm_svm
cm_bag
cm_logr

# Calculating correlation of accuracy and AUC values
cat("Correlation between accuracy rates and areas under curves: " , round(cor(acc_vec, auc_vec),2))


# Assessing classification methods with different performance measures 
# by using optimal threshold values of each method
vec_all = c() 
for (i in c(1:5)){
  if (i==1){a = table(dat_log[valid_ind,]$PainDiagnosis, pred_logReg > opt_t_logr)
    threshold = opt_t_logr
    r_name = 'opt_t_logr'
  }  else if (i == 2) {
    a = table(dat[valid_ind,]$PainDiagnosis, y_pred_ROC[,2] > opt_t_rf)
    threshold = opt_t_rf
    r_name = 'opt_t_rf'
  }  else if (i == 3) {
    a = table(dat[valid_ind,]$PainDiagnosis, y_pred_CF[,2] > opt_t_ct)
    threshold = opt_t_ct
    r_name = 'opt_t_ct'
  }  else if (i == 4) {
    a = table(dat[valid_ind,]$PainDiagnosis, attr(y_pred_SVM,"probabilities")[,2] > opt_t_svm)
    threshold = opt_t_svm
    r_name = 'opt_t_svm'
  }  else {
    a = table(dat[valid_ind,]$PainDiagnosis, y_pred_bag$prob[,2] > opt_t_bag)
    threshold = opt_t_bag
    r_name = 'opt_t_bag'
  } 
  
  # Performance measure with optimal threshold
  #threshold = opt_t_rf
  # Sensitivity
  sens = a[4]/(a[2]+a[4])
  # Specificity
  spec = a[1]/(a[1]+a[3])
  # Precision (PPV)
  prec = a[4]/(a[4]+a[3])
  # NPV
  npv = a[1]/(a[1]+a[2])
  # Accuracy
  acc = (a[1]+a[4])/(a[1]+a[3]+a[2]+a[4])
  # FDR
  fdr = a[3]/(a[4]+a[3])
  # FPR
  fpr = a[1]/(a[1]+a[3])
  # F1 Measure
  F1 = 2*prec*sens/(prec+sens)
  #AUC 
  #auc = fpr/sens
  #auc
  vec_opt = c(threshold,sens,spec,prec,npv,acc,fdr,fpr, F1)
  vec_opt = round(vec_opt,2)
  vec_opt
  vec_all = rbind(vec_all,vec_opt)
  rownames(vec_all)[i] = r_name
}
colnames(vec_all) = c("threshold","sens","spec","prec","npv","acc","fdr","fpr","F1")
vec_all # This vector shows all performance measures for each method at once





# Applying best method 'RANDOM FOREST' in terms of accuracy rates with test set
classifier = randomForest(x = dat[train_ind,-1],
                          y = dat[train_ind,]$PainDiagnosis,
                          ntree = 50)
# Predicting the test set results
y_pred_RF = predict(classifier, newdata = dat[test_ind,-1])

# Making the Confusion Matrix and calculating accuracy rate
cm = table(dat[test_ind,]$PainDiagnosis, y_pred_RF)
cm
acc_best = sum(diag(cm))/sum(cm)
cat("Accuracy rate of the best method on test set: " , round(acc_best,3))



# Applying best method 'SVM' in terms of accuracy rates with test set
classifier = svm(formula = PainDiagnosis ~ .,
                 data = dat[train_ind,],
                 type = 'C-classification',
                 kernel = 'linear', probability = TRUE)

# Predicting the test set results
y_pred = predict(classifier,type="prob", newdata = dat[test_ind,-1], probability = TRUE)

# Making the Confusion Matrix and calculating accuracy rate
cm = table(dat[test_ind,]$PainDiagnosis, y_pred)
cm
acc_best = sum(diag(cm))/sum(cm)
cat("Accuracy rate of the best method on test set: " , round(acc_best,3))























######### STAT40590 CLUSTERING PART ###########
# Removing target label
dat_clus = dat[,-1]
head(dat_clus)

dat_clus_catg = dat_clus[, c("Gender","DurationCurrent","PainLocation", "Criterion2", 
             "Criterion4", "Criterion6","Criterion7" ,"Criterion8","Criterion9" , "Criterion10", "Criterion13",
             "Criterion19", "Criterion20", "Criterion26", "Criterion28", "Criterion32", "Criterion33")]

# dat_clus_numer matrix is formed by the numerical variables of the original data without label column
dat_clus_numer = dat_clus[, c("Age","SurityRating", "RMDQ", "vNRS", "SF36PCS", "SF36MCS" , "PF", "BP","GH","VT","MH", "HADSAnx")]

# klaR library for clustering data with categorical variables using k-modes
# install.packages("klaR")
library(klaR)

# finding sum of withindiff values for k 1:10 in k-modes
tot.categ = c()
for (i in 1:10) {
  fitkmo <- kmodes(dat_clus_catg, i, iter.max = 10, weighted = FALSE)
  tot.categ = c(tot.categ,sum(fitkmo$withindiff))
}
plot(1:10, tot.categ, xlab = "k", ylab = "withindiff", main = "k modes with categorical data")



# clustering data with numerical variables using k-means
# finding the within sum of squares values for k 1:10 in k-means
tot.withinss = c()
for (i in 1:10) {
  fitkm <- kmeans(dat_clus_numer, centers=i)
  tot.withinss = c(tot.withinss,fitkm$tot.withinss)
}
plot(1:10, tot.withinss, xlab = "k", ylab = "Total Within SS", main = "the within sum of squares values versus k")



# k-means algorithm with different k values from 2 to 5
par(mfrow = c(2, 2))

# Load useful packages
library(cluster)
dist_slht <- dist(dat_clus_numer, method="euclidean")^2

for (i in c(2,3,4,5)){
  fitkm <- kmeans(dat_clus_numer, centers=i)
  sil <- silhouette(fitkm$cluster,dist_slht)
  silvals<-sil[,3]
  avg = round(mean(silvals), digits = 2)
  hist(silvals, main = paste("silhoutte values with k =",  i,"with mean " ,  avg, sep=" "))
}

par(mfrow = c(1, 1))

# Silhouette plot with k=2
fitkm <- kmeans(dat_clus_numer, centers=2)
sil <- silhouette(fitkm$cluster,dist_slht)
plot(sil)



# Clustering whole data using k-medoids
# The link benefited from https://towardsdatascience.com/clustering-on-mixed-type-data-8bbd0a2569c3
sil_width <- c()
for(i in 2:8){  
  pam_fit <- pam(dat_clus, diss = TRUE, k = i)  
  sil_width[i] <- pam_fit$silinfo$avg.width  
}
plot(1:8, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width", main = "K-medoids with different k values")
lines(1:8, sil_width)




# Comparing estimated partition with the actual diagnoses
# Clustering with k-modes / categorical variables
fitkmo <- kmodes(dat_clus_catg, 2, iter.max = 10, weighted = FALSE)
fitkmo$cluster

# Comparison table for k-modes
table(fitkmo$cluster, as.numeric(dat[,1]))


# Clustering with k-means / numerical variables
fitkm <- kmeans(dat_clus_numer, centers=2)
fitkm$cluster

# Comparison table for k-means
table(fitkm$cluster, as.numeric(dat[,1]))


# Clustering with PAM / using all variables except target label
pam_fit <- pam(dat_clus, diss = TRUE, k = 2)  

# Comparison table for k-medoids
table(pam_fit$clustering, as.numeric(dat[,1]))


# Assessing clustering methods with different performance measures 
vec_clus = c() 
for (i in c(1:3)){
  if (i==1){a = table(fitkmo$cluster, as.numeric(dat[,1]))
  r_name = 'k-modes / categorical data'
  }  else if (i == 2) {
    a = table(fitkm$cluster, as.numeric(dat[,1]))
    r_name = 'k-means / numeric data'
  }  else {
    a = table(pam_fit$clustering, as.numeric(dat[,1]))
    r_name = 'k-medoids / all data'
  }
  # Sensitivity
  sens = a[4]/(a[2]+a[4])
  # Specificity
  spec = a[1]/(a[1]+a[3])
  # Precision (PPV)
  prec = a[4]/(a[4]+a[3])
  # NPV
  npv = a[1]/(a[1]+a[2])
  # Accuracy
  acc = (a[1]+a[4])/(a[1]+a[3]+a[2]+a[4])
  # FDR
  fdr = a[3]/(a[4]+a[3])
  # FPR
  fpr = a[1]/(a[1]+a[3])
  # F1 Measure
  F1 = 2*prec*sens/(prec+sens)
  #AUC 
  #auc = fpr/sens
  #auc
  vec_opt = c(sens,spec,prec,npv,acc,fdr,fpr, F1)
  vec_opt = round(vec_opt,2)
  vec_opt
  vec_clus = rbind(vec_clus,vec_opt)
  rownames(vec_clus)[i] = r_name
}
colnames(vec_clus) = c("sens","spec","prec","npv","acc","fdr","fpr","F1")
vec_clus # This vector shows all performance measures for each method at once

