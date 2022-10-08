z.prop <- function(x1,x2,n1,n2){
  numerator = (x1/n1) - (x2/n2)
  p.common = (x1+x2) / (n1+n2)
  denominator = sqrt(p.common * (1-p.common) * (1/n1 + 1/n2))
  z.prop.ris = numerator / denominator
  #print(c(z.prop.ris - qnorm(0.975)*denominator, z.prop.ris + qnorm(0.975)*denominator))
  return(z.prop.ris)
}

z.prop(12, 30, 12+35, 30+8)
prop.test(x = c(12, 30), n = c(12+35, 30+8), correct = FALSE) # non-parametric

RR <- 0.034 / 0.016

2 * pf(4.2/10.3, 30-1, 23-1) # p-vale for variance equality test
#var.test(a,b)

x1bar <- 26 
x2bar <- 15
v1 <- 4.2
v2 <- 10.3
n1 <- 30
n2 <- 23
t.stat <- (x1bar - x2bar) / (sqrt(v1/n1 + v2/n2))
#t.test(Control,Treat,alternative="less")

(130 - 90) - 1.96 * 4.47
(130 - 90) + 1.96 * 4.47

2*pnorm((168-197)/sqrt(168+197), lower.tail = TRUE)
2*pnorm(-1.518, lower.tail = TRUE)
mcnemar.test(matrix(c(12, 197, 168, 67), nrow = 2, byrow=T), correct = FALSE)
1-pchisq((197-168)^2/(197+168),1)

mcnemar.test(matrix(c(12, 168, 197, 67), nrow = 2, byrow=T), correct = TRUE)
(168-197)/sqrt(197+168)
#chisq.test()

bb <- c(3,7,2,8,2)
ab <- c(2,5,4,5,4)
db <- bb - ab
mean(db)
sd(db)
t.test(bb, ab, paired = TRUE)#, alternative='greater')
t.test(db)#, alternative='greater')

1-pt(2.61,18)
2*(1-pt(1.85,19))

which.max(c(25/83, 18/49, 36/105))
round(105*70/237)

chisq.test(matrix(c(9,	5,	9, 6,	9,	5, 7,	11,	8), byrow=T, nrow=3), correct=FALSE)

CramerV <- sqrt(chisq.test(matrix(c(11, 11, 22, 9, 8, 20), nrow=3, byrow=T), correct=FALSE)$statistic / ((11+11+22+9+8+20)*(2-1)))
sqrt(10.584 / ((11+11+22+9+8+20)*(2-1)))

sum(c(1.85,	1.58,	-2.84,
      1.46,	1.22,	-2.21,
      -2.78,	-2.35,	4.23))

chisq.test(matrix(c(7, 3, 1, 12), byrow=T, nrow=2), correct=FALSE)

chisq.test(c(15, 20, 19), p=c(0.1, 0.4, 0.5), correct=F)


chisq.test(matrix(c(5, 5, 7, 3), byrow=T, nrow=2), correct=F)


3000 - 7.3*50
r = -0.5
sx = 4.86
sy = 3.50
r*sy/sx
#200 = a - 0.8*20
a = 0.8*20 + 200 
rss <- sum((c(23,16,10,15)-c(16.8,18.4,15.2,13.6))^2) # residual SS
tss <- sum((c(23,16,10,15) - mean(c(23,16,10,15)))^2)
1 - rss / tss # r squared
b = 7.2 
se = 2.8
tstat = b / se
n = 20
pt(tstat, df=n-2, lower.tail=FALSE)

n=100
b=-10.3
se=6.06
b-qt(0.025, df=n-2, lower.tail=FALSE)*se
b - 1.99*se 
b+qt(0.025, df=n-2, lower.tail=FALSE)*se
b + 1.99*se

4.2 * (1.03)^6


rss <- sum((c(2.88,3.22,3.56,3.9,4.24)- mean(c(3.8,2,4,3,5)))^2) # regression SS
tss <- sum((c(3.8,2,4,3,5) - mean(c(3.8,2,4,3,5)))^2)
rss / tss # r squared

ess <- 10.6 # error / residual SS
tss <- 26.2
rss <- tss - ess # regression SS
fstat <- (rss / 4) / (ess / (20 - 5))
#pf(fstat, df1=4, df2=20-5, lower.tail=FALSE)
qf(0.05, df1=4, df2=20-5, lower.tail=FALSE)

b = 4.5
se = 2.5
tstat = b / se
b - qt(0.025, df=30 - 3, lower.tail=FALSE) * se
b + qt(0.025, df=30 - 3, lower.tail=FALSE) * se

#log(p / (1-p)) = 0.4 + 1.166*0.5 = 0.983
p = 1 / (1 +exp(-0.983))

8/9
9/11

bluecow <- c(3,5,4,7)
popstar <- c(3,6,5,8)
demon <- c(2,4,5,6)
sum((bluecow - mean(bluecow))^2 + (popstar - mean(popstar))^2 + (demon - mean(demon))^2) / (12-3) # within MS
sum(var(bluecow) + var(popstar) + var(demon)) / 3

grandmean <- 4.83
sum(4*(mean(bluecow)-grandmean)^2 + 4*(mean(popstar)-grandmean)^2 + 4*(mean(demon)-grandmean)^2) / (3-1) # between MS

(25-10)+qt(0.025, df=15-3, lower.tail=FALSE)*sqrt((102/(15-3))*(1/5+1/5))
(25-10)-qt(0.025, df=15-3, lower.tail=FALSE)*sqrt((102/(15-3))*(1/5+1/5))

msbet <- 36.5
mswth <- 20.6
fstat <- msbet / mswth
g <- 3
n <- 10
pf(fstat, df1=g-1, df2=n-g, lower.tail = FALSE)

pf(3.2, df1=1, df2=65, lower.tail = FALSE) # factorial main
pf(5.6, df1=1, df2=56, lower.tail = FALSE) # factorial interaction

fstat <- 6.2
emss <- 6.8 # ms resid = mswth
msbet <- fstat * emss

wilcox.test(c(3.5, 4.2, 2.3, 3.0, 1.0) - 2, alternative='greater', exact=TRUE)
#sum(c(4.2, 3.5, 3.0, 2.3, 1.0) - 2) # 1 + 2.5 + 4 + 5

wilcox.test(c(2.5, 7.4, 7.2, 6.5), c(8.0, 5.5, 3.2, 6.2), exact=TRUE, paired=TRUE)
      # ranks 1    7    6     5       8      3   2   4
      # sum                  19                     17

wilcox.test(c(2.5, 7.4, 7.2, 6.5) - c(8.0, 5.5, 3.2, 6.2), exact=TRUE)

#kruskal-wallis
pchisq(10.2, df=4-1, lower.tail=FALSE)

cor(c(10,13,3,16,4), c(1,4,2,3,5), method='spearman')
cor.test(c(10,13,3,16,4), c(1,4,2,3,5), method='spearman')

library(tseries)
x <- as.factor(c('C', 'D', 'C', 'D', 'D', 'C', 'C', 'C', 'C', 'D', 'D', 'C', 'D'))  # randomness
runs.test(x)

# practice test
r = 0.097
sy = 5.054
sx = 4.989
r*sy/sx

rss = 305.868
tss = 12258.407
rss / tss

2*pt(1.304 / 0.471, df=481-2, lower.tail=FALSE)
2*pt(0.096 / 0.046, df=481-2, lower.tail=FALSE)

rss = 16.117
k = 2 + 1
ess = 12242.290
n = 481
fstat = rss / (k-1) / (ess / (n-k))
pf(fstat, df1=k-1, df2=n-k, lower.tail = FALSE)

#rsq / (1-rsq) = 0.13
rsq = 1/ (1 + 1/0.13)

chisq.test(matrix(c(62,	82,	68, 98,	79,	92), byrow=T, nrow=2), correct=FALSE)

pt(1.304/0.471, df=480, lower.tail = FALSE)
pt(0.096/0.046, df=480, lower.tail = FALSE)


ssb <- 35*(34.543 - 33.714)^2 + 35*(32.686 - 33.714)^2 + 35*(33.914 - 33.714)^2
ssw <- 35 * 5.792^2 + 35 * 5.034^2 + 35 * 5.136^2
fstat <- (ssb / (3 - 1)) / (ssw / (105 - 3))

11.452 / 2

p1 <- 163 / (163 + 138)
p2 <- 106 / (106 + 74)
n1 <- 163 + 138
n2 <- 106 + 74
p <- (163 + 106) / (n1 + n2)
se <- sqrt(p*(1-p)*(1/n1+1/n2))
zstat <- (p1 - p2) / se
pnorm(zstat, 0.05)

#online statistics education rice

30.0037 + 1.30363*0 + 0.0959*36


x1bar <- 35.505 
x2bar <- 35.667
v1 <- 5.021^2
v2 <- 4.948^2
n1 <- 301
n2 <- 180
t.stat <- (x1bar - x2bar) / (sqrt(v1/n1 + v2/n2))

n <- 160
p <- 51 / n
p0 <- 0.217 
z.stat <- (p - p0) / sqrt(p0*(1-p0)/n)

mcnemar.test(matrix(c(203, 226, 106, 46), nrow = 2, byrow=T), correct = TRUE)
(226-106)/sqrt(226+106)

2*pt(2.790, 479, lower.tail=FALSE)

#kruskal-wallis statistic
12 * (160*(241-240.05)^2+161*(241-265.92)^2+160*(241-216.94)^2) / (481*482)

# final
(29.319 - 31.848) / sqrt(10.3808^2/254 + 11.9676^2/254)
# (y - 70.84) / 20.094 = 0.335 * (x - 2.6733) / 0.7534
# 20.094 / 0.7534
0.335 * 20.094 / 0.7534

fstat <- (520.579 / 3) / (92852.744 / 257) 
pf(fstat, 3, 257, lower.tail=FALSE)
pf((9523.244 / 3) / (92852.744 / 257), 3, 257, lower.tail=FALSE)
pf((1184.582 / 1) / (92852.744 / 257), 1, 257, lower.tail=FALSE)

zstat <- (227/293 - 92/143) / sqrt((319/436)*(1-319/436)*(1/293+1/143))
(227/293 - 92/143) - 2 * sqrt((319/436)*(1-319/436)*(1/293+1/143))
(227/293 - 92/143) + 2 * sqrt((319/436)*(1-319/436)*(1/293+1/143))

(186/297 - 71/147) / sqrt((257/444)*(1-257/444)*(1/297+1/147))

9053.624/58462.259

# Q21
between <- 9*(81.11111-63.48148)^2+9*(45.44444-63.48148)^2+9*(63.88889-63.48148)^2
within <- (9-1)*18.31969^2 + (9-1)*14.99259^2 + (9-1)*20.96094^2
fstat <- (between / (3-1)) / (within / (27-3))

busEcon <- c(87, 88, 60, 92, 73, 103, 96, 85.30529, 45.69470)
law <- c(47, 57, 30, 25, 54, 55, 36, 70.56238, 34.43758)
socSci <- c(55, 73, 74, 57, 63, 46, 98, 82.27139, 26.72862)
type <- rep(c("be", "law", "ss"), c(9, 9, 9))
tempFrame <- data.frame(score=c(busEcon, law, socSci), type=type)
tapply(tempFrame$score, tempFrame$type, FUN=mean) ## matches inputs
tapply(tempFrame$score, tempFrame$type, FUN=sd) ## matches inputs
summary(aov(score ~ type, data=tempFrame)) ## F-statistic not accepted


# Q22
2*pnorm((81 - 64) / sqrt(81 + 64), lower.tail = FALSE)
mcnemar.test(matrix(c(97, 81, 64, 158), nrow = 2, byrow=T), correct = FALSE)
pchisq((81 - 64)^2 / (81 + 64),1,lower.tail=FALSE)

zstat <- (222/400 - 239/400) / sqrt((319/436)*(1-319/436)*(1/293+1/143))

# kruskal-wallis 
(12 / (36*(36+1))) * (12*(26.00-18.5)^2+9*(7.611-18.5)^2+15*(19.033-18.5)^2)

(2.727 / 2) / (114.637 / 244)

0.375^2

sqrt(8275.689/58449.514)

chisq.test(matrix(c(51,	15,	37, 206,	39,	80), byrow=T, nrow=2), correct=FALSE)

# Q28
proc <- 2.7479 + 0.77183
logodds <- -2.472 + 0.672 * 1 + 0.397 * proc
prob <- 1 / (1 + exp(-logodds))

# Q29
spooled <- sqrt(((12-1)*1.51213^2+(13-1)*1.26706^2)/(12+13-2))
tstat <- (5.1313 - 6.1078) / (spooled*sqrt(1/12 + 1/13))
pt(tstat, df=12+13-2)