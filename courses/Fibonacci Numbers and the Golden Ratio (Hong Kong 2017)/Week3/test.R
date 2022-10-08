setwd('C:/work/analytics/R/session.coord/')

library(ggplot2)

continued.fraction <- function(f, n) {
  
  res <- c()
  for (i in 1:n) {
    int <- as.integer(f)
    frac <- f - int
    res <- c(res, int)
    f <- 1 / frac
  }
  print(paste('[', res[1], ';', paste(res[2:length(res)], collapse = ','), ',...]', sep=''))
  return(res)
}

approximate.fraction <- function(f) {
  
  res <- c(f[1])
  n <- length(f)
  for (i in 2:n) {
    s <- f[i]
    for (j in seq(i-1, 1, -1)) {
      s <- f[j] + 1 / s
    }
    res <- c(res, s)
  }
  return(res)
}

f <- continued.fraction(sqrt(2), 20)
v <- approximate.fraction(f)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)

ggplot(df[2:nrow(df),], aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

f <- continued.fraction(sqrt(3), 20)
v <- approximate.fraction(f)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)

ggplot(df, aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  #scale_y_log10() +
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

f <- continued.fraction(exp(1), 20)
v <- approximate.fraction(f)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)

ggplot(df, aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  #scale_y_log10() +
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

#options(digits=20)

f <- continued.fraction(pi, 20)
v <- approximate.fraction(f)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)

ggplot(df, aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  #scale_y_log10() +
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

fib <- function(n) {
  f <- rep(1, n)
  for (i in 3:n) {
    f[i] <- f[i-1] + f[i-2]
  }
  return(f)
}

f <- continued.fraction((sqrt(5)+1)/2, 20)
v <- approximate.fraction(f)
F <- fib(21)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)
df$ratio <- F[2:21] / F[1:20] 

ggplot(df, aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  #scale_y_log10() +
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

p <- 2
f <- continued.fraction(1-(sqrt(p)-1)/2, 20)
v <- approximate.fraction(f)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)
df

ggplot(df, aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  #scale_y_log10() +
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

f <- continued.fraction(1-(sqrt(5)-1)/2, 20)
v <- approximate.fraction(f)
F <- fib(21)
df <- data.frame(num.digits=1:length(f), digits=f, approximate.value=v)
df$ratio <- F[2:21] / F[1:20] 

ggplot(df, aes(num.digits, approximate.value)) + 
  geom_point(col='red', size=2) + geom_line(col='blue') + 
  #scale_y_log10() +
  xlab('Continued Fraction Digits') + ylab('Approximate Value')

sunflower.plot <- function() {
  
  p <- 2
  alpha <- 1 - (sqrt(p)-1)/2 #(pi - 3)
  dangle <- 2*pi*alpha
  vel <- 1
  dt <- 1
  angle <- 0
  df <- data.frame(x=0, y=0, angle=angle)
  n <- 500
  df <- rbind(df, data.frame(x=0, y=0, angle=angle))
  for (i in 1:n) {
    print(i)
    df$x <- df$x + vel*cos(df$angle)*dt
    df$y <- df$y + vel*sin(df$angle)*dt
    angle <- angle + dangle
    if (angle > 2*pi) {
      angle <- angle - 2*pi
    }
    df <- rbind(df, data.frame(x=0, y=0, angle=angle))
  }
  p <- ggplot(df, aes(x, y, col=sqrt(x^2+y^2), fill=sqrt(x^2+y^2))) + geom_point(size=4, shape=23) +
    scale_colour_gradient(low='green', high='yellow') + 
    scale_fill_gradient(low='green', high='yellow') + 
    #theme_bw() +
    theme(panel.background = element_rect(fill = "black"),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.border = element_blank(),
          panel.background = element_blank(),
          axis.title.x=element_blank(),
          axis.text.x=element_blank(),
          axis.ticks.x=element_blank(),
          axis.title.y=element_blank(),
          axis.text.y=element_blank(),
          axis.ticks.y=element_blank())  +
    guides(colour=FALSE) +
    guides(fill=FALSE)
  print(p)
  
  library(animation)
  saveGIF({
    for (i in 1:n) {
      print(i)
      df$x <- df$x + vel*cos(df$angle)*dt
      df$y <- df$y + vel*sin(df$angle)*dt
      angle <- angle + dangle
      if (angle > 2*pi) {
        angle <- angle - 2*pi
      }
      df <- rbind(df, data.frame(x=0, y=0, angle=angle))
      p <- ggplot(df, aes(x, y, col=sqrt(x^2+y^2), fill=sqrt(x^2+y^2))) + geom_point(size=4, shape=23) +
        scale_colour_gradient(low='green', high='yellow') + 
        scale_fill_gradient(low='green', high='yellow') + 
        #theme_bw() +
        theme(panel.background = element_rect(fill = "black"),
              axis.line = element_line(colour = "black"),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.border = element_blank(),
              panel.background = element_blank(),
              axis.title.x=element_blank(),
              axis.text.x=element_blank(),
              axis.ticks.x=element_blank(),
              axis.title.y=element_blank(),
              axis.text.y=element_blank(),
              axis.ticks.y=element_blank())  +
        guides(colour=FALSE) +
        guides(fill=FALSE)
      print(p)
    }
  }, 'test.gif', interval=0.01)
}

seive.prime <- function(n) {
  
  primes <- 2:n
  for (i in 2:sqrt(n)) {
    j <- 2
    while (i*j <= n) {
      index <- which(primes == i*j)
      if (length(index) > 0) {
        primes <- primes[-index]
      }
      j <- j + 1
    }
  }  
  return(primes)
}


test.num <- function() {
  
  primes <- seive.prime(1000)
  ratios <- c()
  for (p in primes) {
    N <- c()
    N1 <- 10 #100
    for (k in 1:N1) {
      n <- (((1+sqrt(p))/2)^k - ((1-sqrt(p))/2)^k)/sqrt(p)
      N <- c(N, n)
    }
    N
    #print(N[1:(length(N)-1)]/N[2:length(N)])
    ratios <- c(ratios, N[99]/N[100])
  }
  print(ratios)
  plot(primes, ratios, type='l')
  df <- data.frame(primes=primes, ratios=ratios)
  ggplot(df, aes(primes, ratios)) + geom_point() + geom_line() +
    scale_x_continuous(breaks=primes) +
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
}