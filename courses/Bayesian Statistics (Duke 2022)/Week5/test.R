Now, let's use `BAS` with the *Zellner-Siow Cauchy* prior on the coefficients instead, using the following code.

```{r}
movies_numeric <- as.data.frame(movies_no_na)
for (col in names(movies_numeric)) {
  print(col)
  print(class(movies_numeric[col]))
  if (is.factor(movies_numeric[col])) {
    movies_numeric[col] <- as.integer(movies_numeric[col])     
    #movies_numeric[col] <- ifelse(movies_numeric[col] == 'yes', 1, 0)
  }
}
zs_lscore <- bas.lm(audience_score ~ ., 
                data = movies_numeric,
                prior = "ZS-null",
                modelprior = uniform(),   
                method = "MCMC",
                MCMC.iter = 1000
              )
zs_lscore
summary(zs_lscore)
coef_lscore <- coefficients(zs_lscore)
diagnostics(coef_lscore)
```

#options(warnings= -1)



