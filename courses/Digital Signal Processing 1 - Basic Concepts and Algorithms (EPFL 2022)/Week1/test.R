x <- function(n) {
  ifelse(n %in% 1:3, (-1)^n*n, 0)
}

y <- function(n) {
  sum(x(n-7*(-100:100)))
}

X <- c()
for (n in -1000:1000) {
  X <- c(X, x(n))
}

Y <- c()
for (n in -100:100) {
  Y <- c(Y, y(n))
}

plot(1:length(X), X, pch=19)
plot(1:length(Y), Y, pch=19)