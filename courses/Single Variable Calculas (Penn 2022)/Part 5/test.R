epsilon <- 0.01
inf <- 10^5

limit <- function(f, v, se, left=TRUE, n=100) {
  if (left) {
    start <- se
    end <- v + epsilon
  } else {
    start <- v + epsilon
    end <- se
  }
  step <- (end - start) / n
  seqs <- seq(start, end, step)
  if (!left) {
    seqs <- rev(seqs)
  }
  values <- c()
  for (i in seqs) {
    values <- c(values, f(i))
  }
  return(values)
}

f <- function(x) {
  #return (log(1 + x) / x)
  return (1/(1+x))^(1+x)
}

#limit(f, 0, 1, left=FALSE)
#limit(f, inf, 10)

forward_diff <- function(an) {
  values <- c()
  for (i in 1:(length(an)-1)) {
    values <- c(values, an[i+1] - an[i])
  }
  return(values)
}

forward_diff_n <- function(an, max_iter=5) {
  print(paste(c(0, paste(an, collapse=',')), collapse=': '))
  for (i in 1:max_iter) {
    an <- forward_diff(an)
    print(paste(c(i, paste(an, collapse=',')), collapse=': '))
    if (all(an == 0)) {
      return(c(TRUE, i))
    }
  }
  return(FALSE)
}

#forward_diff(2^(1:10))
forward_diff_n(c(0,2,8,18,32,50,72))
#forward_diff_n(c(0,2,6,14,30,62,126))

inf <- 25 #100#00

test_convergence <- function(f) {
  #print(c(f(inf), (f(inf))^(1/inf), f(inf) / f(inf-1), sum(f(1:inf)), sum(f(1:(2*inf))), sum(f(1:(10*inf))), sum(f(1:(100*inf))), sum(f(1:(1000*inf)))))
  print(c(f(inf), (f(inf))^(1/inf), f(inf) / f(inf-1), sum(f(1:inf)), sum(f(1:(2*inf))), sum(f(1:(10*inf))), sum(f(1:(20*inf))), sum(f(1:(50*inf)))))
  plot(1:inf, f(1:inf), type='l')
}

#test_convergence(function(n) ((-1)^n*2^(1/n))) 
#test_convergence(function(n) ((2^(1/n)))) 
test_convergence(function(n) ((-1)^n*n^2*2^n/factorial(n)))
test_convergence(function(n) (n^2*2^n/factorial(n)))

#1
#test_convergence(function(n) (3*(n-1)/(3*n))^(n^2)) # exp(-1/3)
#test_convergence(function(n) ((n+1)/(2*n-1))^n)
#test_convergence(function(n) (2*atan(n)/pi)^n)
#test_convergence(function(n) (log(3*n+1)/n)^(2*n))
#test_convergence(function(n) ((n^4*(log(cos(1/n)))^2)/(n+1)))
#test_convergence(function(n) ((2*n^2+4*n+1)/(3*n^2+1))^n)
#test_convergence(function(n) (n/log(n))^(n/2))
#test_convergence(function(n) (pi/(2*atan(n)))^n)
#test_convergence(function(n) (3*n/(3*n-1))^(n^2))
#test_convergence(function(n) (2*n/(2*n+1))^(2*n)) 

inf <- 10#00
#3
#test_convergence(function(n) ((3^n)/(n^3)))
#test_convergence(function(n) ((cos(n))^2/(n^2)))
#test_convergence(function(n) ((n^2-1)/((n^3+n^2+1))))
#test_convergence(function(n) (1+1/n^2)^(n^3))
#test_convergence(function(n) (1/(2^n+n^2)))
#test_convergence(function(n) ((1+(-1)^n)/(2+(-1)^n)))
#test_convergence(function(n) ((n^3+1)/((n^10+1)^(1/3))))
#test_convergence(function(n) ((n-1)/(n+1))^(n^2+n))
#test_convergence(function(n) ((n^2)/(2*n+1))^(n))
#test_convergence(function(n) (log((n^2+1)/n^2)))
#test_convergence(function(n) ((2*n)/(3*n+1))^n)
#test_convergence(function(n) ((n*(n^2+1)^2)/(n^8+2*n^2+4*n+1)))
test_convergence(function(n) (factorial(n)/factorial(2*n)))

#test_convergence(function(n) (3*(n-1)/(3*n))^(n^2))

test_convergence(function(n) (log(cos(1/n)))^2)
test_convergence(function(n) (sin(1/n)))
test_convergence(function(n) (log(1+1/n)))
test_convergence(function(n) (n^2*tan((1/n)^3)))
test_convergence(function(n) ((n+2)/n^2)^n)
test_convergence(function(n) ((-1)^n*(n+2)/n^2)^n)
test_convergence(function(n) ((3*n-1)/n^2)^n)
test_convergence(function(n) (factorial(n)^2/factorial(2*n)))


f1 <- function(N) {
  fs <- c()
  for (n in N) {
    f <- 1
    for (i in 1:n) {
      f <- f * (2*i-1) / i
    }
    fs <- c(fs, f)
  }
  return(fs)
}
#f1 <- Vectorize(f1)
test_convergence(f1)

f2 <- function(N) {
  fs <- c()
  for (n in N) {
    f <- 1
    for (i in 1:n) {
      f <- f * asin(i/(i+1))
    }
    fs <- c(fs, f)
  }
  return(fs)
}
test_convergence(f2)

f3 <- function(N) {
  fs <- c()
  for (n in N) {
    f <- 1
    for (i in 1:n) {
      f <- f + prod((1:i)^2+1) / prod((1:i)^3+1)
    }
    fs <- c(fs, f)
  }
  return(fs)
}
#f1 <- Vectorize(f1)
test_convergence(f3)

#test_convergence(function(n) ((3*n-1)/n^2)^n)
test_convergence(function(n) ((3*n-1)/n^2)^n)

(1/0.00005-1)/2
#(1/0.00005)^(1/3)-1
S <- sum(1/(1:10000)^3)
s <- 0
for (n in 1:1000) {
  s <- s + 1/n^3
  print(c(n, abs(S-s)))
  if (abs(S - s) < 0.00005) {
    break
  }
}

for (n in 1:25) {
  E <- exp(-n-1)*(n+1)
  print(E)
  if (E < 0.00001) {
    print(n)
    break
  }
}

n <- 1:10
(-1)^n/factorial(2*n)

S <- sum((-1)^(1:20)/(factorial(2*(1:20))))
s <- 0
for (n in 1:100) {
  s <- s + (-1)^n/factorial(2*n)
  print(c(n, abs(S-s)))
  if (abs(S - s) < 1e-6) {
    break
  }
}

Euler <- function(f, h, tn, xn, mit=10) {
  print(c(tn, xn))
  for (i in 1:mit) {
    xn <- xn + h*f(xn, tn)
    tn <- tn + h
    print(c(tn, xn))
  }
}

f <- function(xn, tn) {
  return (tn - 2*xn)
}

Euler(f, 0.1, 0, 3)

integrate(function(x) log(x) - exp(1)*log(x)/x, 1, exp(1))
integrate(function(x) 1/(x^2-1), 2, Inf)
integrate(function(x) pi*(5-x)^2, 2, 5)$value / pi

n <- seq(20,100,0.01)
plot(n, 1/n, type='l')
lines(n, sin(1/n), col='red')

integrate(function(x) sin(1/x), 1, 10000)
integrate(function(x) 1/sqrt(x*(x+2)), 1, 10000)
