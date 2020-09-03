###################################
# Julian Cabezas Pena
# Big data Analysis and Project
# Project 1 1
# Support Vector Machines
####################################


if (!require(tidyverse)) {install.packages("tidyverse")}

library(tidyverse)
library(gridExtra)
library(grid)

cv_primal <- read.csv("./Cross_Validation/cost_cv_svm_primal.csv")
cv_dual <- read.csv("./Cross_Validation/cost_cv_svm_dual.csv")
cv_sklearn <- read.csv("./Cross_Validation/cost_cv_svm_sklearn.csv")

cv_primal$Implementation <- "Primal form"
cv_dual$Implementation <- "Dual form"
cv_sklearn$Implementation <- "Scikit-Learn"

cv <- rbind(cv_primal,cv_dual,cv_sklearn)

p <-ggplot(sv, aes(x=cost, y=accuracy*100, group=Implementation)) +
  geom_line(aes(color=Implementation,linetype=Implementation))+
  geom_point(aes(color=Implementation)) +
  scale_color_manual(values=c("dodgerblue4","red","black")) +
  ylim(96,98)+
  ylab("Mean Accuracy (%)") +
  xlab("Cost (C) value") +
  theme_bw(12)

p

ggsave("./Document_latex/CV_cost.pdf",p)

# Histogram of model parameters

para_primal <- read.csv("./Results/model_parameters_primal.csv")
para_dual <- read.csv("./Results/model_parameters_dual.csv")
para_sklearn <- read.csv("./Results/model_parameters_sklearn.csv")

para_primal$Implementation <- "Primal form"
para_dual$Implementation <- "Dual form"
para_sklearn$Implementation <- "Scikit-Learn"

para <- rbind(para_primal[1:200,],para_dual[1:200,],para_sklearn[1:200,])

p <- ggplot(para, aes(x=w_b, fill=Implementation)) + geom_histogram(alpha=0.2, position="identity")

p <-ggplot(para, aes(x=Implementation, y=w_b, color=Implementation)) +
  geom_boxplot() +
  scale_color_manual(values=c("dodgerblue4","red","black")) +
  ylab("Feature weight(w)") +
  theme_bw(12)

ggsave("./Document_latex/weights.pdf",p)

