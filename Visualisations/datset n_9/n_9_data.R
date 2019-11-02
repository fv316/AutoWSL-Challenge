setwd("C:/Users/fvice/Downloads/autowsl_starting_kit/starting_kit/code_submission")

df <- read.csv("C:/Users/fvice/Desktop/lx_courses/DAUL/n_9.csv", header = TRUE)
df2 <- df[12:14]

library(FactoMineR)

summary(df)
dim(df)
View(df)
rownames(df)
plot(df$n_1, df$n_2)
plot(df2)

#### PCA
res.pca= PCA(df)
plot.PCA(res.pca, choix = "ind", cex = 0.7)
plot.PCA(res.pca, select = "contrib 10")

#### Visualize the missing data pattern
library(VIM)
aggr(df)
aggr(df,only.miss=TRUE,numbers=TRUE,sortVar=TRUE)
res <- summary(aggr(df,prop=TRUE,combined=TRUE))$combinations
res[rev(order(res[,2])),]

