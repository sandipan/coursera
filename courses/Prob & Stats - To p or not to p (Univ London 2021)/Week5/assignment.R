x <- c(992, 1002, 1000, 1001, 998, 999, 1000, 995, 1003, 1001, 997, 997)
n <- length(x)
n
xbar <- mean(x)
xbar
mu <- 1000
mu
sigma <- sqrt(1.5)
sigma

(998.75-1000)/(1.224745/sqrt(12))
Z = (xbar - mu) / (sigma / sqrt(n))
Z
2*pnorm(Z)


#curve(dnorm, from=-4, to=4, 
#      main = "The Standard Normal Distibution", 
#      ylab = "Probability Density",
#      xlab = "X")

#polygon_points <- list(x = polygon_x, y = polygon_y)
#polygon(polygon_points, col = col)

library(ggplot2)
z <- seq(-6,6,0.01)
fz <- dnorm(z)
q <- Z #qnorm(0.1) # the quantile
x <- seq(-6, q, 0.01)
y <- c(dnorm(x), 0, 0)
x <- c(x, q, -6)
x2 <- seq(-q, 6, 0.01)
y2 <- c(dnorm(x2), 0, 0)
x2 <- c(x2, 6, -q)
ggplot() + geom_line(aes(z, fz)) +
           geom_polygon(data = data.frame(x=x, y=y), aes(x, y), fill='red', col='red') + 
           geom_polygon(data = data.frame(x=x2, y=y2), aes(x, y), fill='red', col='red') +
           geom_vline(xintercept = q, col='red') +
           geom_vline(xintercept = -q, col='red') +
           ylab('')

#2*pnorm(-(0.44-0.4)/(sqrt(0.4*0.6/1000)))      



# generates data
library(gamlss)
x<-rEXP(2000,mu=3)
y<-rGB2(2000,mu=1, sigma=3, nu=8, tau=2)
df <- data.frame(x, y)
hist(x,200)
hist(y,200)
median(x)
median(y)

# quantile gam regression
library(qgam)
library(mgcViz)
library(ks)
library(DAMisc)
q0 <- qgamV(y ~ s(x), data=df, qu=0.5)
q0_pred <- predict(q0,se.fit=TRUE)
df$q0_p <- q0_pred[["fit"]]

# kernel density estimation
q0_kde <- data.frame(cbind(df$x, df$y))
q0_hpi <- Hpi.kfe(q0_kde) #or Hscv
q0_fhat <- ks::kde(q0_kde, H=q0_hpi, gridsize = 500)
dimnames(q0_fhat[['estimate']]) <- list(q0_fhat[["eval.points"]][[1]], q0_fhat[["eval.points"]][[2]])
library(reshape2)
q0_melt <- melt(q0_fhat[['estimate']])

library(ggplot2)
ggplot() +
  geom_point(data = df, aes(x = x, y = y), color="grey80", size=1) +
  scale_x_continuous(limits=c(-1,30),expand=c(0,0),breaks=c(-1,0,10,20,30),labels=c(-1,0,10,20,30),trans=yj_trans(0.1))+
  scale_y_continuous(limits=c(0.5,7),expand=c(0,0),breaks=seq(0.5,6.5,1),labels=seq(0.5,6.5,1),trans=yj_trans(0.1))+
  geom_line(aes(x=x,y=q0_p),data=df,colour="red3",size=1.4)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["99%"], color="red", size=0.7)+ #1% centroid
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["90%"], color="black", size=0.7)+ #10%
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["80%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["70%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["60%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["50%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["40%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["30%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["20%"], color="black", size=0.7)+
  geom_contour(q0_melt, mapping=aes(x=Var1, y=Var2, z=value), breaks=q0_fhat[["cont"]]["10%"], color="black", size=0.7) #90%