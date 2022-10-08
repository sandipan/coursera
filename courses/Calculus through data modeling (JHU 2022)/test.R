ReimannSum <- function(f, a, b, n) {
  I <- integrate(f, a, b)
  h <- (b-a)/n
  L <- sum(h*f(a+(0:(n-1))*h))
  R <- sum(h*f(a+(1:n)*h))
  return(c(L, I$value, R)) 
}

for (n in c(5, 10, 100, 1000)) {
  print(ReimannSum(function(x) x^2, 0, 1, n))
}

ReimannSum(function(x) 4*x^2, a=0, b=8, n=4)
ReimannSum(function(x) 4*x-3, a=2, b=5, n=3)
#ReimannSum(function(x) 7+6*x-x^2, a=-1, b=7, n=3)
ReimannSum(function(x) 7+6*x-x^2, a=0, b=3, n=3)
ReimannSum(function(x) x/3-2, a=0, b=9, n=1000)

f <- function(x) {
  if (x == 0) {
    return (182.9)
  } else if (x == 10) {
    return (168)
  } else if (x == 20) {
    return (106.6)
  } else if (x == 30) {
    return (99.8)
  } else if (x == 40) {
    return (124.5)
  } else if (x == 50) {
    return (176.1)
  } else if (x == 60) {
    return (175.6)
  }
}

vf <- Vectorize(f)
x <- seq(0, 50, 10)
sum(10*unlist(vf(x))/3600)

f <- function(x) {
  if (x %in% c(0.2,6.2)) {
    return (1.3)
  } else if (x %in% c(1.7, 4.7)) {
    return (4.128)
  } else if (x == 3.2) {
    return (5.3)
  }
}

vf <- Vectorize(f)
x <- seq(1.7, 6.2, 1.5)
sum(1.5*unlist(vf(x)))

integrate(function(x) (x^4-2)/x^2, 4, 6)
#integrate(function(x) 4, -1, 2)

5*pi^2-20
integrate(function(x) -10*x, -pi, 0)$value + integrate(function(x) -10*sin(x), 0, pi)$value
integrate(function(x) (x+4)*(x-1)^4, 0, 1)
integrate(function(x) x^2*sqrt(4-x^2), 0, 2)
integrate(function(x) (3*x+1)^sqrt(2), 0, 1)

library(rSymPy)  
x <- Var("x")
sympy("integrate(exp(x)/(4-exp(2*x)))") 
sympy("integrate(sqrt(2*x**2-3)/x**2)")
