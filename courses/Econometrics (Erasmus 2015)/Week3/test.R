setwd('C:\\courses\\Coursera\\Current\\Econometrics\\Week3')

df <- read.csv('TrainExer 3-5.csv')

df <- df[-1]

m <- lm(LogEqPrem~BookMarket+Inflation+EarnPrice+DivPrice+NTIS, data=df)
m.sum <- summary(m)
res <- rbind(round(m.sum$coef[,1],4), round(m.sum$coef[,4],4), round(m.sum$r.squared,3))
rownames(res) <- c('coeff', 'p.val', 'R^2')
which.max(res[2,2:ncol(res)])

m <- lm(LogEqPrem~BookMarket+Inflation+EarnPrice+DivPrice, data=df)
m.sum <- summary(m)
res <- rbind(round(m.sum$coef[,1],4), round(m.sum$coef[,4],4), round(m.sum$r.squared,3))
rownames(res) <- c('coeff', 'p.val', 'R^2')
which.max(res[2,2:ncol(res)])

m <- lm(LogEqPrem~BookMarket+EarnPrice+DivPrice, data=df)
m.sum <- summary(m)
res <- rbind(round(m.sum$coef[,1],4), round(m.sum$coef[,4],4), round(m.sum$r.squared,3))
rownames(res) <- c('coeff', 'p.val', 'R^2')
which.max(res[2,2:ncol(res)])

m <- lm(LogEqPrem~BookMarket+EarnPrice, data=df)
m.sum <- summary(m)
res <- rbind(round(m.sum$coef[,1],4), round(m.sum$coef[,4],4), round(m.sum$r.squared,3))
rownames(res) <- c('coeff', 'p.val', 'R^2')
which.max(res[2,2:ncol(res)])

m <- lm(LogEqPrem~BookMarket, data=df)
m.sum <- summary(m)
res <- rbind(round(m.sum$coef[,1],4), round(m.sum$coef[,4],4), round(m.sum$r.squared,3))
rownames(res) <- c('coeff', 'p.val', 'R^2')

unrestricted <- lm(LogEqPrem~BookMarket+Inflation+EarnPrice+DivPrice+NTIS, data=df)
restricted <- lm(LogEqPrem~BookMarket, data=df)

unrestricted.n <- nrow(df)
unrestricted.sse <- sum(resid(unrestricted) ^2)

#unrestricted.n + unrestricted.n * log(2*pi) + unrestricted.n*log(unrestricted.sse/unrestricted.n) + log(unrestricted.n) * (6+1) # BIC
#unrestricted.n + unrestricted.n * log(2*pi) + unrestricted.n*log(unrestricted.sse/unrestricted.n) + 2 * (6+1) # AIC

summary(unrestricted)$r.squared
AIC(unrestricted, k=2)
BIC(unrestricted) #AIC(unrestricted, k=log(nrow(df)))
sum(resid(unrestricted) ^2)

summary(restricted)$r.squared
AIC(restricted, k=2)
BIC(restricted) #AIC(restricted, k=log(nrow(df)))
sum(resid(restricted) ^2)

F = ((sum(resid(restricted) ^2) - sum(resid(unrestricted) ^2)) / 4) /  (sum(resid(unrestricted) ^2) / (nrow(df) - 6))  # H0: restricted
pf(1,nrow(df)-6, 4)
waldtest(unrestricted, restricted)
anova(unrestricted, restricted)
#coeftest(unrestricted)

