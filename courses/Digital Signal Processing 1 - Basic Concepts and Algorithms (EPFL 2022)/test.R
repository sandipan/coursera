library(matlib)

v1 <- rep(1/2, 4)
v2 <- c(rep(1/2,2),rep(-1/2,2))
v1 %*% v2

v3 <- rep(c(1/2,-1/2), 2)
v4 <- c(1/2,-1/2,-1/2,1/2) #rep(1, 4)

for (x1 in c(-1/2,1/2)) {
  for (x2 in c(-1/2,1/2)) {
    for (x3 in c(-1/2,1/2)) {
      for (x4 in c(-1/2,1/2)) {
        v4 <- c(x1, x2, x3, x4)
        B <- matrix(c(v1, v2, v3, v4), nrow=4)
        D <- det(B)
        if (D != 0) {
          crossprod(B)
          B1 <- GramSchmidt(B) #, verbose=TRUE))
          if (all(B == B1)) {
            print(D)
            print(B1)
          }
          zapsmall(crossprod(B1))
        }
      }
    }
  }  
}

y <- c(-1/2,1/2,-3/2,-1/2)
B <- matrix(c(v1, v2, v3, v4), nrow=4)
alpha <- solve(B, y)
rowSums(t(alpha*t(B))) #sum(alpha[k]*B[,k])

det(matrix(c(y, v1, v2, v4), nrow=4))
det(matrix(c(y, v2+v3, v3, v4), nrow=4))
det(matrix(c(y, v2, v3, v4), nrow=4))

A <- matrix(c(0,0,0,1,1,0,0,0,0,1,0,0,0,0,1,0), nrow=4, byrow=T)
A%*%A%*%A%*%A



Gram_Schmidt <- function(A) {
  n <- ncol(A)
  Q <- 0*A
  R <- matrix(rep(0, n*n), nrow=n)
  for (j in 1:n) {
    v <- A[,j]
    if (j > 1)
      for (i in 1:(j-1)) {
        R[i, j] <- t(Q[,i]) %*% A[,j]
        v <- v - R[i,j] * Q[,i]
      }
    R[j,j] <- sqrt(v %*% v)
    Q[,j] <- v / R[j,j]
  }
  return(list(Q=Q, R=R))
}

Gram_Schmidt_QR <- function(A) {
  res <- qr(A)
  return(list(Q=qr.Q(res), R=qr.R(res)))
}

A <- matrix(c(4,3,-2,1), nrow=2)
Gram_Schmidt(A)
Gram_Schmidt_QR(A)
library(matlib)
GramSchmidt(A)

library(ggplot2)
library(microbenchmark)
autoplot(microbenchmark(Gram_Schmidt(A),
                        Gram_Schmidt_QR(A),
                        GramSchmidt(A), times=1000L))

library(gbm)
set.seed(101) # for reproducibility
N <- 1000
X1 <- runif(N)
X2 <- 2 * runif(N)
X3 <- ordered(sample(letters[1:4], N, replace = TRUE), levels = letters[4:1])
X4 <- factor(sample(letters[1:6], N, replace = TRUE))
X5 <- factor(sample(letters[1:3], N, replace = TRUE))
X6 <- 3 * runif(N)
mu <- c(-1, 0, 1, 2)[as.numeric(X3)]
SNR <- 10 # signal-to-noise ratio
Y <- X1 ^ 1.5 + 2 * (X2 ^ 0.5) + mu
sigma <- sqrt(var(Y) / SNR)
Y <- Y + rnorm(N, 0, sigma)
X1[sample(1:N,size=500)] <- NA # introduce some missing values
X4[sample(1:N,size=300)] <- NA # introduce some missing values
data <- data.frame(Y, X1, X2, X3, X4, X5, X6)
# Fit a GBM
s = "Jacaranda.puberula"
set.seed(102) # for reproducibility
gbm1 <- gbm(Y ~ ., data = data, var.monotone = c(0, 0, 0, 0, 0, 0),
            distribution = "gaussian", shrinkage = 0.1,
            n.trees = ifelse(s == "Jacaranda.puberula" | s == "Cestrum.intermedium", 50, 10000),
            interaction.depth = 3, bag.fraction = 0.5, train.fraction = 0.5,
            n.minobsinnode = 10, cv.folds = 5, keep.data = TRUE,
            verbose = FALSE, n.cores = 1)

sqrt(2)*exp(pi/4*1i)
sqrt(2)*exp(-pi/4*1i)
sqrt(2)*exp(-3*pi/4*1i)

plot(1:100, sin(1:100))

N <- 64
y3 <- rep(1,N)
y1 <- 2*cos(2*pi/64*4*(0:(N-1)))
y2 <- 0.5*sin(2*pi/64*8*(0:(N-1)))
plot(0:(N-1), y1, type='l', col='blue')
lines(0:(N-1), y2, col='green')
lines(0:(N-1), y3, col='red')

plot(0:(N-1), fft(y1), col='blue', pch=19, ylim=c(-20,70))
points(0:(N-1), Im(fft(y2)), col='green', pch=19)
points(0:(N-1), fft(y3), col='red', pch=19)

2*16^2 + 3*64^2

N <- 5
x <- rep(c(-1,1), N)
x <- rep(c(1,-1), N)
fft(x)

x <- 1:5
fft(fft(x))

N <- 100
n <- -N:N #0:(N-1)
x <- 1/5 + cos(pi/2*n) + cos(pi*n)/3  #cos(2*pi/3*n) #cos(1*n) #
ws <- seq(-pi, pi, 0.01)
Xs <- c()
for (w in ws) {
  Xs <- c(Xs, sum(x[n+N+1]*exp(-n*w*1i))) 
}
n <- 0:10
x <- 1/5 + cos(pi/2*n) + cos(pi*n)/3  #cos(2*pi/3*n) #cos(1*n) #
plot(n, x, col='blue', pch=19)
plot(ws, abs(Xs), col='blue', pch=19)

n <- seq(-pi, pi, 0.01)
x <- c()
for (i in n){
  if ((i <= pi/8) & (i >= -pi/8)) {
    x <- c(x, 1)
  } else {
    x <- c(x, 0)
  }
}
plot(n, x, col='blue', pch=19)
#plot(n, sin(5*pi/8)*x, col='blue', pch=19)


library(dismo)
data(Anguilla_train)
# reduce data set to speed things up a bit
Anguilla_train = Anguilla_train[1:200,]
angaus.tc5.lr01 <- gbm.step(data=Anguilla_train, gbm.x = 3:14, gbm.y = 2, family = "bernoulli",
                            max.trees = ifelse(s == "Jacaranda.puberula" | s == "Cestrum.intermedium", 50, 10000),
                            tree.complexity = 5, learning.rate = 0.01, bag.fraction = 0.5)


library(mlbench)
data(BostonHousing)

set.seed(102) # for reproducibility
gbm1 <- gbm(medv ~ ., data = BostonHousing, #var.monotone = c(0, 0, 0, 0, 0, 0),
            distribution = "gaussian", shrinkage = 0.1,
            n.trees = 100,
            interaction.depth = 3, bag.fraction = 0.5, train.fraction = 0.5,
            n.minobsinnode = 10, cv.folds = 5, keep.data = TRUE,
            verbose = FALSE, n.cores = 1)
predict(gbm1)

gbm2 <- gbm.step(data=BostonHousing, gbm.x = 1:13, gbm.y = 14, family = "gaussian",
                            max.trees = 10000,
                            tree.complexity = 5, learning.rate = 0.01, bag.fraction = 0.5)



predict(gbm2)

library(tidyverse)
data.frame(index=1:nrow(BostonHousing),
           actual=BostonHousing$medv, pred.gbm1=predict(gbm1), pred.gbm2=predict(gbm2)) %>% gather('pred', 'medv', -index) %>%
  ggplot(aes(index, medv, group=pred, color=pred)) + geom_point() + geom_line()