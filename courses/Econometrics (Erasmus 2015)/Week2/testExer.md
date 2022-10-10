Test Exercise 2: Answers to the Questions
========================================================

This test exercise is of a theoretical nature. In our discussion of the \(F\)-test, the total set of explanatory factors was
split in two parts. The factors in \(X_1\) are always included in the model, whereas those in \(X_2\) are possibly removed.
In questions \((a)\), \((b)\), and \((c)\) you derive relations between the two **OLS** estimates of the effects of \(X_1\) on \(y\), one in the large model and the other in the small model. In parts \((d)\), \((e)\), and \((f)\), you check the relation of question \((c)\) numerically for the wage data of our lectures. 


We use the notation of Lecture 2.4.2 and assume that the standard regression assumptions \(A1-A6\) are satisfied for
the unrestricted model. The restricted model is obtained by deleting the set of \(g\) explanatory factors collected in
the last \(g\) columns \(X_2\) of \(X\). We wrote the model with \(X = (X_1 X_2)\) and corresponding partitioning of the **OLS**
estimator \(b\) in \(b_1\) and \(b_2\) as \(y = X_1\beta_1 + X_2\beta_2 + \epsilon = X_1b_1 + X_2b_2 + e\). We denote by \(b_R\) the **OLS** estimator of \(\beta_1\) obtained by regressing \(y\) on \(X_1\), so that \(b_R = (X_1^{\prime}X_1)^{-1}X_1^{\prime} y\). Further, let \(P = (X_1^{\prime} X_1)^{-1}X_1^{\prime} X_2\).


* (a) Prove that \(E(b_R) = \beta_1 + P\beta_2\).

+ Proof: 

\(\begin{align*}
&E(b_R) \\ 
&= E((X_1^{\prime}X_1)^{-1}X_1^{\prime} y) \\
&= E((X_1^{\prime}X_1)^{-1}X_1^{\prime} (X_1\beta_1 + X_2\beta_2 + \epsilon)) \\
&= E((X_1^{\prime}X_1)^{-1}(X_1^{\prime}X_1)\beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}X_2\beta_2 + (X_1^{\prime}X_1)^
{-1}X_1^{\prime}\epsilon) \\
&= E(I.\beta_1 + P.\beta_2 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}.\epsilon) \\
&= E(\beta_1) + P.E(\beta_2) + (X_1^{\prime}X_1)^{-1}X_1^{\prime}.E(\epsilon) \\
&= \beta_1 + P\beta_2 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}.0 \\
&= \beta_1 + P\beta_2
\end{align*} \)

* (b) Prove that \(var(b_R) = \sigma^2(X_1^{\prime} X_1)^{-1} \).

+ Proof:

\(\begin{align*}
&var(b_R) \\ 
&= E((b_R-E(b_R))(b_R-E(b_R))^{\prime}) \\
&= E((b_R-\beta_1 - P\beta_2)(b_R-\beta_1 - P\beta_2)^{\prime}) \\
&= E(((X_1^{\prime}X_1)^{-1}X_1^{\prime} y -\beta_1 - (X_1^{\prime} X_1)^{-1}X_1^{\prime} X_2\beta_2)((X_1^{\prime}X_1)^{-1}X_1^{\prime} y -\beta_1 - (X_1^{\prime} X_1)^{-1}X_1^{\prime} X_2\beta_2)^{\prime}) \\
&= E((((X_1^{\prime}X_1)^{-1}X_1^{\prime}).(y - X_2\beta_2) - \beta_1)(((X_1^{\prime}X_1)^{-1}X_1^{\prime}).(y - X_2\beta_2) -\beta_1)^{\prime})  \\
&= E((((X_1^{\prime}X_1)^{-1}X_1^{\prime}).(X_1\beta_1 + \epsilon)) - \beta_1)(((X_1^{\prime}X_1)^{-1}X_1^{\prime}).(X_1\beta_1 + \epsilon)) -\beta_1)^{\prime}) \\
&= E(((X_1^{\prime}X_1)^{-1}.(X_1^{\prime}X_1).\beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon - \beta_1).((X_1^{\prime}X_1)^{-1}.(X_1^{\prime}X_1).\beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon - \beta_1)^{\prime}) \\
&= E((I.\beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon - \beta_1).(I.\beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon - \beta_1)^{\prime}) \\
&= E((\beta_1 - \beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon).(\beta_1 - \beta_1 + (X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon )^{\prime})  \\
&= E(((X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon).((X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon )^{\prime}) \\
&= E((X_1^{\prime}X_1)^{-1}X_1^{\prime}\epsilon\epsilon^{\prime}X_1(X_1^{\prime}X_1)^{-1}) \\
&= (X_1^{\prime}X_1)^{-1}X_1^{\prime}E(\epsilon\epsilon^{\prime})X_1(X_1^{\prime}X_1)^{-1} \\
&= (X_1^{\prime}X_1)^{-1}X_1^{\prime}\sigma^2X_1(X_1^{\prime}X_1)^{-1} \\
&= \sigma^2(X_1^{\prime}X_1)^{-1}X_1^{\prime}X_1(X_1^{\prime}X_1)^{-1} \\
&= \sigma^2(X_1^{\prime}X_1)^{-1}
\end{align*} \)

* (c) Prove that \(b_R = b_1 + Pb_2\).

+ Proof:

\(\begin{align*}
&S(b_1, b_2) = e^{\prime}e = (y - X_1b_1 - X_2b_2)^{\prime}(y - X_1b_1 - X_2b_2) \\ 
&=y^{\prime}y - y^{\prime}(X_1b_1 + X_2b_2) - b_1^{\prime}X_1^{\prime}y + b_1^{\prime}X_1^{\prime}X_1b_1 + b_1^{\prime}X_1^{\prime}X_2b_2 - b_2^{\prime}X_2^{\prime}y + b_2^{\prime}X_2^{\prime}X_1b_1 + b_2^{\prime}X_2^{\prime}X_2b_2 \\
&=y^{\prime}y - (y^{\prime}X_1b_1 + b_1^{\prime}X_1^{\prime}y) - (y^{\prime}X_2b_2 + b_2^{\prime}X_2^{\prime}y) + b_1^{\prime}X_1^{\prime}X_1b_1 + b_1^{\prime}X_1^{\prime}X_2b_2 + b_2^{\prime}X_2^{\prime}X_1b_1 + b_2^{\prime}X_2^{\prime}X_2b_2 \\
&=y^{\prime}y - 2b_1^{\prime}X_1^{\prime}y - 2b_2^{\prime}X_2^{\prime}y + b_1^{\prime}X_1^{\prime}X_1b_1 + b_1^{\prime}X_1^{\prime}X_2b_2 + b_2^{\prime}X_2^{\prime}X_1b_1 + b_2^{\prime}X_2^{\prime}X_2b_2 \\
&\Rightarrow \frac{\delta{S}}{\delta{b1}}=-2X_1^{\prime}y+2(X_1^{\prime}X_1b_1)+X_1^{\prime}X_2b_2+b_2^{\prime}X_2^{\prime}X_1 = -2X_1^{\prime}y+2(X_1^{\prime}X_1b_1)+2(X_1^{\prime}X_2b_2)=0 \\
&\Rightarrow X_1^{\prime}y=X_1^{\prime}X_1b_1+X_1^{\prime}X_2b_2 \\
&\Rightarrow (X_1^{\prime}X_1)^{-1}X_1^{\prime}y=b_1+(X_1^{\prime}X_1)^{-1}X_1^{\prime}X_2b_2\\
&\Rightarrow b_R = b_1 + Pb_2
\end{align*} \)


* Now consider the wage data of Lectures \(2.1\) and \(2.5\). Let \(y\) be log-wage \((500\times 1)\) vector, and let \(X_1\) be the \((500\times 2)\) matrix for the constant term and the variable 'Female'. Further let \(X_2\) be the \((500 \times 3)\) matrix with observations of the variables 'Age', 'Educ' and 'Parttime'. The values of \(b_R\) were given in Lecture \(2.1\), and those of \(b\) in  Lecture \(2.5\).

* (d) Argue that the columns of the \((2 × 3)\) matrix \(P\) are obtained by regressing each of the variables 'Age', 'Educ',
and 'Parttime' on a constant term and the variable 'Female'.

+ Let \(X_2(i)\) (which is a \((500\times 1)\) vector) denote the \(i^{th}\) column of the matrix \(X_2\). Then, from lecture \(2.1\), we have,


\(\begin{align*}
X_2(1) &= Age = 40.05 - 0.11Female + e = X_1.(40.05\; -0.11)^T + e \\
X_2(2) &= Educ = 2.26 - 0.49Female + e = X_1.(2.26\; -0.49)^T + e \\
X_2(3) &= Parttime = 0.20 + 0.25Female + e =  X_1.(0.20\; 0.25)^T + e\\
\end{align*}\)

\(\Rightarrow X_2 = (X_2(1)\;X_2(2)\;X_2(3)) 
= X_1.\left( \begin{array}{ccc}
40.05 & 2.26 & 0.20\\
-0.11 & -0.49 & 0.25 \\
\end{array} \right) + e = X_1.P + e \), since \(P=(X_1^{\prime}X_1)^{-1}.(X_1^{\prime}X_2)\).

* (e) Determine the values of \(P\) from the results in Lecture \(2.1\).

+ From \((d)\), we have, \(P=\left( \begin{array}{ccx}
40.05 & 2.26 & 0.20 \\
-0.11 & -0.49 & 0.25 \\
\end{array} \right)\).

* (f) Check the numerical validity of the result in part \((c)\). Note: This equation will not hold exactly because the
coefficients have been rounded to two or three decimals; preciser results would have been obtained for higher
precision coefficients.

+ From lecture \(2.5\) again we have, \(log(Wage) = 3.05 - 0.04Female + 0.03Age + 0.23Educ - 0.37Parttime + e \Rightarrow b_1=\left( \begin{array}{c}
3.05 \\
-0.04
\end{array} \right)\; b_2=\left( \begin{array}{c}
0.03 \\
0.23 \\
-0.37
\end{array} \right) \Rightarrow P.b_2=\left( \begin{array}{ccc}
40.05 & 2.26 & 0.20 \\
-0.11 & -0.49 & 0.25 \\
\end{array} \right).\left( \begin{array}{c}
0.03 \\
0.23 \\
-0.37
\end{array} \right)=\left( \begin{array}{c}
1.6473 \\
-0.2085
\end{array} \right) \Rightarrow b_1+Pb_2 = \left( \begin{array}{c}
4.6973 \\
-0.2485
\end{array} \right)\). But, from lecture \(2.1\) we have, \(log(Wage) = 4.73 - 0.25Female + e \Rightarrow b_R=\left( \begin{array}{c} 4.73 \\
-0.25
\end{array} \right) \). Hence, we can see \(b_R \approx b_1 + Pb_2 \) (as per note, some loss of precision happens because of rounding).

 

```
## [1] "P.b2"
```

```
##         [,1]
## [1,]  1.6473
## [2,] -0.2085
```

```
## [1] "b1+P.b2"
```

```
##         [,1]
## [1,]  4.6973
## [2,] -0.2485
```
 
