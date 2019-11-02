setwd("C:/Users/fvice/Downloads/autowsl_starting_kit/starting_kit/code_submission")

Ecolo <- read.csv("C:/Users/fvice/Desktop/lx_courses/DAUL/s_7.csv", header = TRUE)

library(FactoMineR)

summary(Ecolo)
dim(Ecolo)
View(Ecolo)
rownames(Ecolo)
colnames(Ecolo)
coltime <- match("t_1",colnames(Ecolo))
Ecolo2 <- Ecolo[-coltime]
cols <- c("c_1","c_2","c_3","c_4","c_5","c_6","c_7","c_8","c_9","c_10","c_11","c_12","c_13","c_14","c_15")
colindex <- match(cols,colnames(Ecolo2))
plot(Ecolo$n_1, Ecolo$n_2)
#plot(Ecolo2)

#### PCA
res.pca= PCA(Ecolo)
res.pca <- PCA(Ecolo2, quali.sup = colindex) # default option to deal with NA is mean imputation
plot.PCA(res.pca, choix = "ind", cex = 0.7)
plot.PCA(res.pca, select = "contrib 10")

#### Visualize the missing data pattern
library(VIM)
aggr(Ecolo)
aggr(Ecolo,only.miss=TRUE,numbers=TRUE,sortVar=TRUE)
res <- summary(aggr(Ecolo,prop=TRUE,combined=TRUE))$combinations
res[rev(order(res[,2])),]

mis.ind <- matrix("o",nrow=nrow(Ecolo),ncol=ncol(Ecolo))
mis.ind[is.na(Ecolo)] <- "m"
dimnames(mis.ind) <- dimnames(Ecolo)
library(FactoMineR)
resMCA <- MCA(mis.ind)
plot(resMCA,invis="ind",title="MCA graph of the categories")
