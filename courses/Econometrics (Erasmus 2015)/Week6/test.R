setwd('C:/courses/Coursera/Current/Econometrics/Week6')
df <- read.csv('TrainExer65.csv')
library(tseries)
r <- adf.test(df$LOGIP, alternative = "stationary", k = 2)
library(fUnitRoots)
unitrootTest(df$LOGIP, lags = 2, type = "ct", title = NULL, description = NULL)
r <- adfTest(df$LOGIP, lags = 2, type = "ct", title = NULL, description = NULL)
summary(r@test$lm)
r <- adfTest(df$LOGCLI, lags = 2, type = "ct", title = NULL, description = NULL)
summary(r@test$lm)
library(quantmod)
df$t <- 3:(2+nrow(df))
df$LOGIPt_1 <- Lag(df$LOGIP,1)[,1]
df$GIPt_1 <- Lag(df$GIP,1)[,1]
df$GIPt_2 <- Lag(df$GIP,2)[,1]
m <- lm(GIP~t+LOGIPt_1+GIPt_1+GIPt_2, df)
df$LOGCLIt_1 <- Lag(df$LOGCLI,1)[,1]
df$GCLIt_1 <- Lag(df$GCLI,1)[,1]
df$GCLIt_2 <- Lag(df$GCLI,2)[,1]
m <- lm(GCLI~t+LOGCLIt_1+GCLIt_1+GCLIt_2, df)