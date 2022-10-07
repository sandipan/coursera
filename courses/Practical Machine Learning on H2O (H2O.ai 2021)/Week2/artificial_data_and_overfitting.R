## Step One is to create a data set, as we saw in earlier videos. If you took my 
## example and modified it, add a comment explaining what your modification was.

set.seed(321)

N <- 1000                                         # number of samples
d <- data.frame(id = 1:N)
d$workEnvironment <- sample(1:5, N, replace=TRUE) # on a scale of 1-5, 1 being bad and 5 being good
v <- round(rnorm(N, mean=60000, sd=20000))        # 68% are 40-80k
v <- pmax(v, 20000)
v <- pmin(v, 100000) #table(v)
d$pay <- v
d$flexibility <- sample(1:5, N, replace=TRUE)     # on a scale of 1-5, 1 being bad and 5 being good
d$managerRel <- sample(1:5, N, replace=TRUE)      # on a scale of 1-5, 1 being bad and 5 being good
d$age <- round(runif(N, min=20, max=60))
head(d)

v <- 125 * (d$pay/1000)^2         # e.g., job satisfaction score is proportional to square of pay (hypothetically)
v <- v + 250 / log(d$age)         # e.g., inversely proportional to log of age 
v <- v + 5 * d$flexibility 
v <- v + 200 * d$workEnvironment
v <- v + 1000 * d$managerRel^3
v <- v + runif(N, 0, 5000)
v <- 100 * (v - 0) / (max(v) - min(v)) # min-max normalization to bring the score in 0-100
d$jobSatScore <- round(v)               # Round to nearest integer (percentage)

## Step Two is to start h2o, and import your data.

library(h2o)
h2o.init()

as.h2o(d, destination_frame = "jobsatisfaction")
jobsat <- h2o.getFrame("jobsatisfaction")

## Step Three is to split the data. If you plan to use cross-validation, split into  train and test. 
## Otherwise split into train, valid and test.

parts <- h2o.splitFrame(
  jobsat,
  ratios = 0.8,
  destination_frames=c("jobsat_train", "jobsat_test"),
  seed = 321)
train <- h2o.getFrame("jobsat_train")  
norw(train)
# 794
test <- h2o.getFrame("jobsat_test")    
norw(test)
# 206 rows

y <- "jobSatScore"
x <- setdiff(names(train), c("id", y))

## Step four is to choose either random forest or gbm, and make a model. It can be 
## classification or regression. Then show the results, on both training data and the test data. 
## You can show all the performance stats, or choose just one (e.g. I focused on MAE in the videos).

# the reasonable model with 10-fold cross-validation
m_res <- h2o.gbm(x, y, train,
              model_id = "model10foldsreasonable",
              ntrees = 20,
              nfolds = 10,
              seed = 123)
h2o.performance(m_res, train = TRUE) # RMSE 2.840688
h2o.performance(m_res, xval = TRUE)  # RMSE 2.973807
h2o.performance(m_res, test)         # RMSE 3.299601

## Step five is then to try some alternative parameters, to build a different model, and show how the results differ.

# overfitting model with 10-fold cross-validation
m_ovf <- h2o.gbm(x, y, train,
              model_id = "model10foldsoverfitting",
              ntrees = 2000,
              max_depth = 20,
              nfolds = 10,
              seed = 123)
h2o.performance(m_ovf, train = TRUE) # RMSE 0.004474786
h2o.performance(m_ovf, xval = TRUE)  # RMSE 0.6801615
h2o.performance(m_ovf, test)         # RMSE 0.4969761
