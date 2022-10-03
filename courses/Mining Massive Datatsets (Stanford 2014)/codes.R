# page rank
# power iteration
r = matrix(c(1,1,1),nrow=3)
A = matrix(c(0,0,1,0.5,0,0,0.5,1,0),nrow=3,ncol=3,byrow=T)
for (i in 1:5) { 
  r = A %*% r
  print(i)
  print(r)
}

# topic specific page rank
b = 0.8
r = matrix(rep(1/4,4))
M = matrix(c(0,1/2,1/2,0,
             1,0,0,0,
             0,0,0,1,
             0,0,1,0),nrow=4,ncol=4,byrow=F)
for (i in 1:5) { 
  print(i)
  print(r)
  r = b * M %*% r + (1 - b) * matrix(c(1,0,0,0)) / 1 
}

# topic specific page rank
b = 0.7
r = matrix(rep(1/4,4))
M = matrix(c(0,1/2,1/2,0,
             1,0,0,0,
             0,0,0,1,
             0,0,1,0),nrow=4,ncol=4,byrow=F)
for (i in 1:50) { 
  print(i)
  print(r)
  r = b * M %*% r + (1 - b) * matrix(c(2,1,0,0)) / 3 
}

# Jaccard
jaccard <- function(vec1, vec2) {
  and <- sum(vec1 == 1 & vec2 == 1)
  or <- sum(vec1 == 1 | vec2 == 1)
  return(as.factor(paste(as.character(and), "/", as.character(or), sep="")))
}

doc1 <- "ABRACADABRA"
doc2 <- "BRICABRAC"
shingles <- NULL
for (i in 1:(nchar(doc1)-1)) {
  shingles <- c(shingles, substring(doc1, i, i + 1))
}
shingles <- unique(shingles)
print(shingles)
print(length(shingles))
shingles2 <- c()
for (i in 1:(nchar(doc2)-1)) {
  shingles2 <- c(shingles2, substring(doc2, i, i + 1))
}
shingles2 <- unique(shingles2)
print(shingles2)
print(length(shingles2))
shingles <- c(shingles, shingles2)
table(shingles)
shingles <- unique(shingles)
#print(shingles[duplicated(shingles)])
#print(sum(duplicated(shingles)))
print(shingles)
df <- NULL
for (shingle in shingles) {
  df <- rbind(df, data.frame(shingle=shingle, 
                             doc1=ifelse(grepl(shingle, doc1), 1, 0),
                             doc2=ifelse(grepl(shingle, doc2), 1, 0)))
}
shingles <- unique(shingles)
print(jaccard(df$doc1, df$doc2))
print(df)

# dist L1 L2
C1 <- c(0, 0)
C2 <- c(100, 40)
#P <- list(c(53, 18), c(66, 5), c(53, 15), c(61, 10))
P <- list(c(63, 8), c(56, 15), c(61, 8), c(53, 10))
for (p in P) {
  L1 <- c(dist(rbind(p, C1), method="manhattan"),
          dist(rbind(p, C2), method="manhattan"))
  L2 <- c(dist(rbind(p, C1), method="euclidean"),
          dist(rbind(p, C2), method="euclidean"))
  print(p)
  print(paste("L1", L1))
  print(paste("L2", L2))
  print(paste("min L1", which.min(L1), "min L2", which.min(L2)))
}

# POW
numbers <- list()
for (i in 1:5) {
  numbers[[i]] <- c(rep(0, i-1), 1, rep(0, 5-i))
}
#numbers
p <- c()
diff <- list()
for (i in 1:100) {
  tmp <- numbers
  tmp[[1]] <- (numbers[[5]] + numbers[[2]]) / 2
  tmp[[2]] <- (numbers[[1]] + numbers[[3]]) / 2
  tmp[[3]] <- (numbers[[2]] + numbers[[4]]) / 2
  tmp[[4]] <- (numbers[[3]] + numbers[[5]]) / 2
  tmp[[5]] <- (numbers[[4]] + numbers[[1]]) / 2
  for (j in 1:5) {
    diff[[j]] <- numbers[[j]] - numbers[[j%%5 + 1]]
  }
  numbers <- tmp
  #print(numbers)
  #print(diff)
  p <- c(p, diff[[1]][1])
}
plot(p, type='l')

# week4A

# normalize
ratings <- matrix(c(1,2,3,4,5,2,3,2,5,3,5,5,5,3,2), nrow=3, byrow=T)
ratings <- as.data.frame(ratings)
names(ratings) <- c("M","N","P","Q","R")
row.names(ratings) <- c("A", "B", "C")	
row.means <- apply(ratings, 1, mean)
#col.means <- apply(ratings, 2, mean)
ratings <- ratings - row.means
col.means <- apply(ratings, 2, mean)
for (i in 1:(ncol(ratings))) {
  ratings[,i] <- ratings[,i] - col.means[i]
}
#ratings <- ratings - col.means

ratings <- matrix(c(1,0,1,0,1,2,
                    1,1,0,0,1,6,
                    0,1,0,1,0,2), nrow=3, byrow=T)
ratings <- as.data.frame(ratings)
row.names(ratings) <- c("A", "B", "C")
for (alpha in c(0, 0.5, 1, 2)) {
  nratings <- ratings
  nratings[,6] <- alpha*ratings[,6]
  print(paste("alpha", alpha))
  print(nratings)
  print(paste("A B", acos(sum(nratings[1,]*nratings[2,])/sqrt(sum(nratings[1,]^2)*sum(nratings[2,]^2)))))
  print(paste("B C", acos(sum(nratings[2,]*nratings[3,])/sqrt(sum(nratings[2,]^2)*sum(nratings[3,]^2)))))
  print(paste("A C", acos(sum(nratings[1,]*nratings[3,])/sqrt(sum(nratings[1,]^2)*sum(nratings[3,]^2)))))
}

# 2 + 12*alpha^2 # AB
# 1 + 12*alpha^2 # BC
# 4*alpha^2      # CA

# Q1
A <- rbind(data.frame(A=0,B=0,C=1,D=0,E=0,F=1,G=0,H=0),
          data.frame(A=0,B=0,C=0,D=0,E=1,F=0,G=0,H=1),
          data.frame(A=1,B=0,C=0,D=1,E=0,F=1,G=0,H=0),
          data.frame(A=0,B=0,C=1,D=0,E=1,F=0,G=1,H=0),
          data.frame(A=0,B=1,C=0,D=1,E=0,F=0,G=0,H=1),
          data.frame(A=1,B=0,C=1,D=0,E=0,F=0,G=1,H=0),
          data.frame(A=0,B=0,C=0,D=1,E=0,F=1,G=0,H=1),
          data.frame(A=0,B=1,C=0,D=0,E=1,F=0,G=1,H=0))
row.names(A) <- names(A)
D <- as.data.frame(diag(nrow(A))*rowSums(A))
row.names(D) <- names(D) <- names(A)
L <- D - A
print(paste("A:", sum(A), length(A[A!=0])))
print(paste("D:", sum(D), length(D[D!=0])))
print(paste("L:", sum(L), length(L[L!=0])))

# Q2
A <- as.data.frame(matrix(rep(0,36), ncol=6, byrow=T))
A[1,2] <- A[2,1] <- A[1,3] <- A[3,1] <- 1
A[2,4] <- A[4,2] <- A[3,4] <- A[4,3] <- 1
A[2,6] <- A[6,2] <- A[4,5] <- A[5,4] <- 1
A[5,6] <- A[6,5] <- 1
row.names(A) <- names(A) <- 1:6
D <- as.data.frame(diag(nrow(A))*rowSums(A))
row.names(D) <- names(D) <- names(A)
L <- D - A
res <- eigen(L)
#mean(res$vectors[,2])
#which(res$vectors[,2] > 0)
#which(res$vectors[,2] < 0)
#which(res$vectors[,2] == 0)

mean(res$vectors[,5])
which(res$vectors[,5] > 0)
which(res$vectors[,5] < 0)
which(res$vectors[,5] > mean(res$vectors[,5]))
which(res$vectors[,5] < mean(res$vectors[,5]))

# surprise number
surprise <- function(stream) {
  return(sum(table(stream)^2))
}

# AMS
AMS <- function(stream, X1, X2, X3) {
  X1.element <- stream[X1]
  X2.element <- stream[X2]
  X3.element <- stream[X3]
  X1.value <- X2.value <- X3.value <- 1
  n <- length(stream)
  for (i in 1:n) {
    if (i > X1 & stream[i] == X1.element) {
      X1.value <- X1.value + 1
    }
    if (i > X2 & stream[i] == X2.element) {
      X2.value <- X2.value + 1
    }
    if (i > X3 & stream[i] == X3.element) {
      X3.value <- X3.value + 1
    }    
  }
  e1 <- n * (2 * X1.value - 1)
  e2 <- n * (2 * X2.value - 1)
  e3 <- n * (2 * X3.value - 1)
  #print(paste("estimate 1:", e1))
  #print(paste("estimate 2:", e2))
  #print(paste("estimate 3:", e3))  
  #print(paste("avg estimate:", (e1 + e2 + e3) / 3))
  print(paste("median estimate:", median(c(e1, e2, e3))))
}

a <- 1
b <- 2
c <- 3
d <- 4
stream <- c(a, b, c, b, d, a, c, d, a, b, d, c, a, a, b)
#print(surprise(stream))
#AMS(stream, 3, 8, 13)

stream <- c(rep(1:10, 7), 1:5)
#print(surprise(stream))
#AMS(stream, 4, 31, 72)
#AMS(stream, 20, 49, 53)
#AMS(stream, 24, 44, 65)
#AMS(stream, 9, 50, 68)

#AMS(stream, 37, 46, 55)
#AMS(stream, 37, 46, 55)
#AMS(stream, 31, 32, 44)
#AMS(stream, 22, 42, 62)

# week 5B
Q1 <- function(gold, green) {
  
  #print(green)
  #print(gold)

  centroids <- list()
  for (i in 1:(nrow(green))) {
    centroids[[i]] <- c(0)
  }
  for (i in 1:(nrow(gold))) {
    min <- Inf
    minindex <- -1
    for (j in 1:(nrow(green))) {
      dist <- (gold[i,1]-green[j,1])^2 + (gold[i,2]-green[j,2])^2 
      if (dist < min) {
        min <- dist
        minindex <- j
      } 
    }
    centroids[[minindex]] <- c(centroids[[minindex]], i)
  }
  print(centroids)
  newgreen <- NULL
  for (i in 1:(nrow(green))) {
    points <- rbind(gold[centroids[[i]][-1],], green[i,]) #gold[centroids[[i]][-1],] #rbind(gold[centroids[[i]][-1],], green[i,])
    #if (class(points) == "numeric") {
    #  points <- matrix(points, ncol=2)
    #}
    #if (nrow(points) > 0) {
    newgreen <- rbind(newgreen, apply(points, 2, mean))
    #}
  }
  print(newgreen)
  return(newgreen)
}

green <- matrix(c(25, 125, 44, 105, 29, 97, 35, 63, 55, 63, 42, 57, 
                  23, 40, 64, 37, 33, 22, 55, 20), ncol=2, byrow=T)
gold <- matrix(c(28, 145, 65, 140, 50, 130, 38, 115, 55, 118,
                 50, 90, 63, 88, 43, 83, 50, 60, 50, 30), ncol=2, byrow=T)
green <- Q1(gold, green)
green <- Q1(gold, green)


Q2 <- function(yrec, brec) {
  centroids <- matrix(c(5,10,20,5),ncol=2,byrow=T)
  yrec <- cbind(yrec, apply(yrec,1,function(x)(x[1]-centroids[1,1])^2+
                              (x[2]-centroids[1,2])^2))
  yrec <- cbind(yrec, apply(yrec,1,function(x)(x[1]-centroids[2,1])^2+
                              (x[2]-centroids[2,2])^2))
  yrec <- cbind(yrec, apply(yrec,1,function(x)which.min(x[3:4])))
  brec <- cbind(brec, apply(brec,1,function(x)(x[1]-centroids[1,1])^2+
                              (x[2]-centroids[1,2])^2))
  brec <- cbind(brec, apply(brec,1,function(x)(x[1]-centroids[2,1])^2+
                              (x[2]-centroids[2,2])^2))
  brec <- cbind(brec, apply(brec,1,function(x)which.min(x[3:4])))
  print(yrec)
  print(brec)
}

#Q2(matrix(c(7,8,7,5,12,8,12,5),ncol=2,byrow=T),
#   matrix(c(13,10,13,4,16,10,16,4),ncol=2,byrow=T))
#Q2(matrix(c(3,3,3,1,10,3,10,1),ncol=2,byrow=T),
#   matrix(c(15,14,20,14,15,10,20,10),ncol=2,byrow=T))
Q2(matrix(c(3,3,3,1,10,3,10,1),ncol=2,byrow=T),
   matrix(c(13,10,13,4,16,10,16,4),ncol=2,byrow=T))
Q2(matrix(c(3,3,3,1,10,3,10,1),ncol=2,byrow=T),
   matrix(c(15,14,20,14,15,10,20,10),ncol=2,byrow=T))

# BALANCE
BALANCE <- function(queries, bids, budgets) {
  revenue <- 0
  for (i in 1:nchar(queries)) {
    query <- substr(queries,i,i)
    print(query)
    advertisers <- which(bids[query] == 1)
    print("advertisers:")
    print(advertisers)
    if (length(advertisers) == 1) {
      advertiser <- advertisers
    }
    else {
      advertiser <- which.max(budgets[advertisers])
    }
    print(paste("advertiser:", advertiser))
    if (budgets[advertiser] > 0) {
      budgets[advertiser] <- budgets[advertiser] - 1
      revenue <- revenue + 1
    }
    print("remaing budgets:")
    print(budgets)
    print(paste("revenue:", revenue))
  }
}

#bids <- as.data.frame(matrix(c(1,0,1,1),ncol=2,byrow=T))
#names(bids) <- c("x", "y")
#BALANCE("xxyy", bids, c(2,2))

bids <- as.data.frame(matrix(c(1,1,0,1,0,1),ncol=3,byrow=T))
names(bids) <- c("x", "y", "z")
#BALANCE("yxxz", bids, c(2,2))
#BALANCE("xzyz", bids, c(2,2))
#BALANCE("xxxz", bids, c(2,2))
#BALANCE("xyzx", bids, c(2,2))
BALANCE("xyxz", bids, c(2,2))
BALANCE("xyyx", bids, c(2,2))
BALANCE("zzxz", bids, c(2,2))
BALANCE("xxxz", bids, c(2,2))

# set cover
elementsFromSet <- function(sets) {
  elements <- c()
  for (set in sets) {
    elements <- c(elements, strsplit(set, split="")[[1]])
  }
  elements <- unique(elements)
  return(elements)
}

dumbSetCover <- function(sets) {
  elements <- elementsFromSet(sets)
  collection <- c()
  covered <- c()
  i <- 1
  while (i < length(sets) & length(setdiff(elements, covered)) > 0) {
    covered <- unique(c(covered, strsplit(sets[i], split="")[[1]]))
    collection <- c(collection, sets[i])
    i <- i + 1    
  }
  return(collection)
}
simpleSetCover <- function(sets) {
  elements <- elementsFromSet(sets)
  collection <- c()
  uncovered <- elements
  i <- 1
  while (i < length(sets) & length(uncovered) > 0) {
    setElements <- strsplit(sets[i], split="")[[1]]
    if (length(intersect(setElements, uncovered)) > 0) {
      uncovered <- setdiff(uncovered, setElements)
      collection <- c(collection, sets[i])
    }
    i <- i + 1    
  }
  return(collection)
}
sizeSortSets <- function(sets) {
  sizeSortedSetsList <- list()
  for (set in sets) {
    if (nchar(set) > length(sizeSortedSetsList)) {
      sizeSortedSetsList[[nchar(set)]] <- c(set)
    }
    else {
      sizeSortedSetsList[[nchar(set)]] <- c(sizeSortedSetsList[[nchar(set)]], set)
    }
  }
  #print(sizeSortedSetsList)
  #print(rev(unlist(sizeSortedSetsList)))
  sizeSortedSets <- c()
  for (i in length(sizeSortedSetsList):1) {
    sizeSortedSets <- c(sizeSortedSets, sizeSortedSetsList[[i]])
  }
  return(sizeSortedSets)
}
largestFirstSetCover <- function(sets) {
  elements <- elementsFromSet(sets)
  sets <- sizeSortSets(sets)
  collection <- c()
  uncovered <- elements
  i <- 1
  while (i < length(sets) & length(uncovered) > 0) {
    setElements <- strsplit(sets[i], split="")[[1]]
    if (length(intersect(setElements, uncovered)) > 0) {
      uncovered <- setdiff(uncovered, setElements)
      collection <- c(collection, sets[i])
    }
    i <- i + 1    
  }
  return(collection)
}
sets <- c("AB", "BC", "CD", "DE", "EF", "FG", "GH", "AH", "ADG", "ADF")
print(elementsFromSet(sets))
print(dumbSetCover(sets))
dumbSetCoverSize <- length(dumbSetCover(sets))
print(simpleSetCover(sets))
simpleSetCoverSize <- length(simpleSetCover(sets))
sizeSortSets(sets)
print(largestFirstSetCover(sets))
largestFirstSetCoverSize <- length(largestFirstSetCover(sets))
optimumSize <- 4
print(dumbSetCoverSize / optimumSize)
print(simpleSetCoverSize / optimumSize)
print(largestFirstSetCoverSize / optimumSize)

MoorePenrosePseudoInverse <- function(diagonal) {
  inv <- diagonal
  for (i in 1:(length(inv))) {
    if (inv[i,i] != 0) {
      inv[i,i] = 1/inv[i,i]
    }
  }
  return(inv)
}
MoorePenrosePseudoInverse(as.data.frame(matrix(c(1,0,0,0,2,0,0,0,0), ncol=3, byrow=T)))

# CURE
maxMinDist <- function(points, ref) {
  
  mind <- c()
  for (i in 1:nrow(points)) {
    d <- c()
    for (j in 1:nrow(ref)) {
      d <- c(d, (points[i,1] - ref[j,1])^2 + (points[i,2] - ref[j,2])^2)
    }
    mind <- c(mind, min(d))
  }
  #print(mind)
  print(row.names(points[which.max(mind),]))
  return(which.max(mind))
}

points <- as.data.frame(matrix(c(1,6,3,7,4,3,7,7,8,2,9,5), ncol=2, byrow=T))
row.names(points) <- letters[1:6]
ref <- as.data.frame(matrix(c(0,0,10,10),ncol=2,byrow=T))
names(points) <- names(ref) <- c("x", "y")
nxt <- maxMinDist(points, ref)
ref <- rbind(ref, points[nxt,])
points <- points[-nxt,]
nxt <- maxMinDist(points, ref)
ref <- rbind(ref, points[nxt,])
points <- points[-nxt,]
nxt <- maxMinDist(points, ref)
ref <- rbind(ref, points[nxt,])
points <- points[-nxt,]
nxt <- maxMinDist(points, ref)
ref <- rbind(ref, points[nxt,])
points <- points[-nxt,]
nxt <- maxMinDist(points, ref)
ref <- rbind(ref, points[nxt,])
points <- points[-nxt,]
nxt <- maxMinDist(points, ref)
ref <- rbind(ref, points[nxt,])
points <- points[-nxt,]

# prefix indexing
library(hash)
prefixIndex <- function(strings, J) {
  h <- hash()
  for (string in strings) {
    Ls <- nchar(string)
    p <- floor((1.000000000000001 - J) * Ls) + 1
    #print(paste(p, Ls, floor((1 - J) * Ls), string))
    for (i in 1:p) {
      chr <- substring(string, i, i)
      if (!has.key(chr, h)) {
        h[chr] <- c()
      }
      h[chr] <- c(h[[chr]], string)
    }
  }
  print(h)
}

#prefixIndex(c('bcdefghij', 'abcdefghij', 'cdefghij'), 0.9)
prefixIndex(c('abcef', 'acdeg', 'bcdefg', 'adfg', 'bcdfgh', 'bceg', 'cdfg', 'abcd'), 0.8)

# HITS
HITS <- function(L, h, n) {
  for (i in 1:n){
    a <- t(L) %*% h
    a <- a / max(a)
    h <- L %*% a
    h <- h / max(h)
  }
  print(h)
  print(a)
}

h <- rep(1,4)
L <- matrix(c(0,1,1,0,
             1,0,0,0,
             0,0,0,1,
             0,0,1,0),nrow=4,ncol=4,byrow=T)
#h <- rep(1,5)
#L <- matrix(c(0,1,1,1,0,
#              1,0,0,1,0,
#              0,0,0,0,1,
#              0,1,1,0,0,
#              0,0,0,0,0),nrow=5,ncol=5,byrow=T)
n <- 1000
HITS(L, h, n)



# page rank
# power iteration
# final exam basic
r = matrix(c(1,1,1,1),nrow=4)
A = matrix(c(0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0),nrow=4,ncol=4,byrow=T)
for (i in 1:5) { 
  r = A %*% r
  print(i)
  print(r)
}







Q1 <- function(gold, green) { # green -> centroids
  
  #print(green)
  #print(gold)
  
  centroids <- list()
  for (i in 1:(nrow(green))) {
    centroids[[i]] <- c(0)
  }
  for (i in 1:(nrow(gold))) {
    min <- Inf
    minindex <- -1
    for (j in 1:(nrow(green))) {
      dist <- abs(gold[i,1]-green[j,1]) + abs(gold[i,2]-green[j,2])
      if (dist < min) {
        min <- dist
        minindex <- j
      } 
    }
    centroids[[minindex]] <- c(centroids[[minindex]], i)
  }
  print(centroids)
  newgreen <- NULL
  for (i in 1:(nrow(green))) {
    points <- rbind(gold[centroids[[i]][-1],], green[i,]) #gold[centroids[[i]][-1],] #rbind(gold[centroids[[i]][-1],], green[i,])
    #if (class(points) == "numeric") {
    #  points <- matrix(points, ncol=2)
    #}
    #if (nrow(points) > 0) {
    newgreen <- rbind(newgreen, apply(points, 2, mean))
    #}
  }
  print(newgreen)
  return(newgreen)
}

green <- matrix(c(1,1,4,4), ncol=2, byrow=T)
gold <- matrix(c(1,1,
                 2,1,
                 2,2,
                 3,3,
                 4,2,
                 2,4,
                 4,4), ncol=2, byrow=T)
for (i in 1:25) {
  green <- Q1(gold, green)
}


R <- matrix(c(2, 0, 4, 0, 0,		
0,3,1,2,0	,
5,0,0,5,0	,
0, 4, 3,0, 2,
4,0,0,5,	1), ncol=5, byrow=T)
cor(R)

x <- c(0,3,1,2,0)
y <- c(0, 4, 3,0, 2)
sum((x - mean(x))* (y - mean(y))) / (sqrt(sum((x - mean(x))^2)) * sqrt(sum((y - mean(y))^2)))
x <- c(3,1)
y <- c(4, 3)
cor(x,y)

# topic specific page rank
b = 0.8
r = matrix(rep(1/3,3))
M = matrix(c(1/3,1/3,1/3,
             0,1,0,
             0,0,0),nrow=3,ncol=3,byrow=F)
for (i in 1:100) { 
  print(i)
  print(r)
  r = b * M %*% r + (1 - b) * matrix(c(1,1,0)) / 2 
}

b = 0.8
r = matrix(rep(1/4,4))
M = matrix(c(0,1,0,0,
             0,0,1,0,
             0,0,0,1,
             1/4,1/4,1/4,1/4),nrow=4,ncol=4,byrow=F)
for (i in 1:100) { 
  print(i)
  print(r)
  r = M %*% r
}

V <- matrix(c(-0.57,-0.11,-0.57,-0.11,-0.57, 
              -0.09, 0.7, -0.09, 0.7, -0.09), ncol=2, byrow=T)
u1 <- matrix(c(5,0,0,0,0))
u2 <- matrix(c(0,2,0,0,4))
c1 <- t(u1)%*%V
c2 <- t(u2)%*%V
sum(c1*c2)

# Q1
A <- rbind(data.frame(A=0,B=1,C=1,D=1,E=1),
           data.frame(A=1,B=0,C=1,D=1,E=1),
           data.frame(A=1,B=1,C=0,D=1,E=1),
           data.frame(A=1,B=1,C=1,D=0,E=1),
           data.frame(A=1,B=1,C=1,D=1,E=0))
row.names(A) <- names(A)
D <- as.data.frame(diag(nrow(A))*rowSums(A))
row.names(D) <- names(D) <- names(A)
L <- D - A
sum(L^2)

log2 <- function(x) {
  if (x == 0) {
    return(0)
  }
  else {
    return(log(x)/log(2))  
  }
}

entropy <- function(p, f) {
  return(-p/(p + f) * log2(p/(p + f)) - f/(p + f) * log2(f/(p + f)))
}

infoGain <- function(p, f, p1, f1, p2, f2) {
  e <- entropy(p, f)
  print(e)
  e1 <- entropy(p1, f1)
  print(e1)
  e2 <- entropy(p2, f2)
  print(e2)
  ev <- (p1+f1)/(p1+f1+p2+f2)*e1 + (p2+f2)/(p1+f1+p2+f2)*e2
  print(ev)
  print(e - ev)
}

infoGain(5, 5, 5, 1, 0, 4)

# HITS
HITS <- function(L, h, n) {
  for (i in 1:n){
    a <- t(L) %*% h
    a <- a / max(a)
    h <- L %*% a
    h <- h / max(h)
  }
  print(h)
  print(a)
}

h <- rep(1,4)
L <- matrix(c(0,1,0,0,
              0,0,1,0,
              0,0,0,1,
              0,0,0,0),nrow=4,ncol=4,byrow=T)
#h <- rep(1,5)
#L <- matrix(c(0,1,1,1,0,
#              1,0,0,1,0,
#              0,0,0,0,1,
#              0,1,1,0,0,
#              0,0,0,0,0),nrow=5,ncol=5,byrow=T)
n <- 1000
HITS(L, h, n)

# Generalized BALANCE
1*(1-exp(-0.8))
2*(1-exp(-0.6))
3*(1-exp(-0.4))
4*(1-exp(-0.2))

# topic specific page rank
b = 0.8
r = matrix(rep(1/3,3))
M = matrix(c(1/3,1/3,1/3,
             0,1,0,
             0,0,0),nrow=3,ncol=3,byrow=F)
for (i in 1:100) { 
  #print(i)
  #print(r)
  r = b * M %*% r + (1 - b) * matrix(c(1,1,0)) / 2 
}
print(r)

r = matrix(rep(1/3,3))
for (i in 1:100) { 
  r = b * M %*% r + (1 - b) * matrix(c(0,1,0)) / 2 
}
print(r)

r = matrix(rep(1/3,3))
for (i in 1:100) { 
  r = b * M %*% r + (1 - b) * matrix(c(1,0,0)) / 2 
}
print(r)


V <- matrix(c(-0.57,-0.11,-0.57,-0.11,-0.57, 
              -0.09, 0.7, -0.09, 0.7, -0.09), ncol=5, byrow=T)
u1 <- matrix(c(5,0,0,0,0))
u2 <- matrix(c(0,5,0,0,0))
u3 <- matrix(c(0,0,0,0,4))
c1 <- t(u1)%*%t(V)
c2 <- t(u2)%*%t(V)
c3 <- t(u3)%*%t(V)

sum(c1*c2)
sum(c1*c3)
sum(c2*c3)

d <- rbind(c1, c2, c3)
h <- hclust(dist(d))
plot(h)