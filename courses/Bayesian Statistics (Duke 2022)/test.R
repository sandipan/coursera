sum(c(4, 19, 20, 23, 23, 25, 25, 26, 27, 28, 28, 28, 29, 30, 32, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 35, 35, 35, 35, 35, 
35, 36, 36, 36, 36, 37, 37, 37, 38, 38, 39, 40, 40, 41, 41, 45, 47, 47, 47, 49) != 32)
sum(abs(c(4, 19, 20, 23, 23, 25)-23))
sum((c(4, 19, 20, 23, 23, 25)-23)^2)
0.05 /0.95

m0 <- 30
n0 <- 100
s0 <- 10
v0 <- 99
n <- 133
ybar <- 28
s <- 13
mn <- (n*ybar + n0*m0) / (n + n0)
nn <- n0 + n
vn <- v0 + n
sn2 <- (s0^2*v0 + s^2*(n-1) + n0*n/nn*(ybar - m0)^2) / vn
c(mn, nn, sn2, vn)

n <- 22
n0 <- 20
ybar <- 1
s <- 3.6
t <- ybar / (s / sqrt(n))
d <- n - 1
((n+n0)/n0)^0.5*((t^2*n0/(n+n0)+d)/(t^2+d))^((d+1)/2)

n <- 1000
n0 <- 1
Z <- 2.055
((n+n0) / n0)^0.5*exp(-0.5*n/(n+n0)*Z^2)

1-(pnorm(3) - pnorm(-3))^100


library(BAS)

data(bodyfat)
bodyfat.lm = lm(Bodyfat ~ Abdomen, data = bodyfat)

bodyfat.lm.bas = bas.lm(Bodyfat ~ Abdomen, data = bodyfat, prior = "hyper-g-n") #EB-local #BIC

summary(bodyfat.lm)
coefficients(bodyfat.lm.bas)

(pnorm(3) - pnorm(-3))^1000
