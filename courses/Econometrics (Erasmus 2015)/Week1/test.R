setwd('C:\\courses\\Coursera\\Current\\Econometrics\\Week1')

library(ggplot2)
library(dplyr)
# Training 1.1

df <- read.csv('TrainExer11.csv')
head(df)

par(mfrow=c(2,1))
hist(df$Age)
hist(df$Expenditures)
ggplot(df, aes(x=Age, y=Expenditures)) + geom_point()
mean(df$Expenditures)
df1 <- df %>% filter(Age <= 40)
df2 <- df %>% filter(Age > 40)
mean(df1$Expenditure)
mean(df2$Expenditure)

