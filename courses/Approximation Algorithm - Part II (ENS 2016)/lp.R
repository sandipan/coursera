library(Rglpk)

## Simple mixed integer linear program.
## maximize: 3 x_1 + 1 x_2 + 3 x_3
## subject to: -1 x_1 + 2 x_2 + x_3 <= 4
## 4 x_2 - 3 x_3 <= 2
## x_1 - 3 x_2 + 2 x_3 <= 3
## x_1, x_3 are non-negative integers
## x_2 is a non-negative real number
#obj <- c(3, 1, 3)
#mat <- matrix(c(-1, 0, 1, 2, 4, -3, 1, -3, 2), nrow = 3)
#dir <- c("<=", "<=", "<=")
#rhs <- c(4, 2, 3)
#types <- c("I", "C", "I")
#max <- TRUE
#Rglpk_solve_LP(obj, mat, dir, rhs, types = types, max = max)
## Same as before but with bounds replaced by
## -Inf < x_1 <= 4
## 0 <= x_2 <= 100
## 2 <= x_3 < Inf
#bounds <- list(lower = list(ind = c(1L, 3L), val = c(-Inf, 2)),
#               upper = list(ind = c(1L, 2L), val = c(4, 100)))
#Rglpk_solve_LP(obj, mat, dir, rhs, bounds, types, max)

# Week1
obj <- c(10, 5, 4)
mat <- matrix(c(1, 1, 1, 1, 0, -1, -5, 1, -2, 6, -1, 1), nrow = 4, byrow=T)
dir <- c(">=", ">=", ">=", ">=")
rhs <- c(10, 2, 4, 8)
types <- c("C", "C", "C", "C")
max <- FALSE
Rglpk_solve_LP(obj, mat, dir, rhs, types = types, max = max)

obj <- c(10, 2, 4, 8)
mat <- matrix(c(1, 1, -5, 6, 1, 0, 1, -1, 1, -1, -2, 1), nrow = 3, byrow=T)
dir <- c("<=", "<=", "<=")
rhs <- c(10, 5, 4)
types <- c("C", "C", "C")
max <- TRUE
Rglpk_solve_LP(obj, mat, dir, rhs, types = types, max = max)




# Week3 (Facility Location Problem)
#               9 Clients                              7 Facilities  
#               Service Cost                           Facility cost
obj <-        c(1,rep(0, 8),                           
                3,3,3,1,rep(0, 5),
                0,0,2,3,0,2,0,2,1,
                0,0,1,1,rep(0, 5),
                0,0,0,6,2,6,rep(0, 3),
                0,1,0,0,1,9,rep(0, 3),
                0,0,0,1,0,3,1,2,0,
                                                       c(1, 2, 2, 4, 10, 3, 2))
mat <- matrix(c(rep(1, 7),   rep(0, 8*7),              rep(0, 7), 
                rep(0, 7),   rep(1, 7),   rep(0, 7*7), rep(0, 7), 
                rep(0, 2*7), rep(1, 7),   rep(0, 6*7), rep(0, 7), 
                rep(0, 3*7), rep(1, 7),   rep(0, 5*7), rep(0, 7), 
                rep(0, 4*7), rep(1, 7),   rep(0, 4*7), rep(0, 7), 
                rep(0, 5*7), rep(1, 7),   rep(0, 3*7), rep(0, 7), 
                rep(0, 6*7), rep(1, 7),   rep(0, 2*7), rep(0, 7), 
                rep(0, 7*7), rep(1, 7),   rep(0, 7),   rep(0, 7), 
                rep(0, 8*7), rep(1, 7),                rep(0, 7), 
                rep(c(-1, rep(0, 6)), 9),              1, rep(0, 6), 
                rep(c(0,-1, rep(0, 5)), 9),            0, 1, rep(0, 5), 
                rep(c(rep(0, 2), -1, rep(0, 4)), 9),   rep(0, 2), 1, rep(0, 4), 
                rep(c(rep(0, 3), -1, rep(0, 3)), 9),   rep(0, 3), 1, rep(0, 3), 
                rep(c(rep(0, 4), -1, rep(0, 2)), 9),   rep(0, 4), 1, rep(0, 2), 
                rep(c(rep(0, 5), -1, 0), 9),           rep(0, 5), 1, 0, 
                rep(c(rep(0, 6), -1), 9),              rep(0, 6), 1), 
                nrow = 9+7, byrow=T)
dir <- rep(">=", 9+7)
rhs <-          c(rep(1, 9),                           rep(0, 7))
types <- rep("C", 9*7 + 7)
bounds <- list(upper = list(ind = 1:(9*7+7), val = rep(1, 9*7+7)))
max <- FALSE
Rglpk_solve_LP(obj, mat, dir, rhs, bounds=bounds, types = types, max = max)

