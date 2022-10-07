library(h2o)
h2o.init()

set.seed(321)

# Let's say an employee's job satisfaction depends on the work environment, pay, flexibility, relationship with manager and age
N <- 1000
d <- data.frame(id = 1:N)
d$workEnvironment <- sample(1:5, N, replace=TRUE)
v <- round(rnorm(N, mean=60000, sd=20000))   # 68% are 40-80k
v <- pmax(v, 20000)
v <- pmin(v, 100000) #table(v)
d$pay <- v
d$flexibility <- sample(1:5, N, replace=TRUE)
d$managerRel <- sample(1:5, N, replace=TRUE)
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

as.h2o(d, destination_frame = "jobsatisfaction")
jobsat <- h2o.getFrame("jobsatisfaction")

parts <- h2o.splitFrame(
  jobsat,
  ratios = 0.8,
  destination_frames=c("jobsat_train", "jobsat_test"),
  seed = 321)
train <- h2o.getFrame("jobsat_train")  # 900
test <- h2o.getFrame("jobsat_test")    # 100

y <- "jobSatScore"
x <- setdiff(names(train), c("id", y))

# reasonable model
m_res <- h2o.gbm(x, y, train,
              model_id = "model10foldsreasonable",
              ntrees = 75,
              nfolds = 10)
h2o.performance(m_res, train = TRUE) # RMSE 3846.467
h2o.performance(m_res, xval = TRUE)  # RMSE 8060.851
h2o.performance(m_res, test)         # RMSE 5156.933

# overfitting model
m_ovf <- h2o.gbm(x, y, train,
              model_id = "model10foldsoverfitting",
              ntrees = 2000,
              max_depth = 15,
              nfolds = 10)
h2o.performance(m_ovf, train = TRUE) # RMSE 17.39557
h2o.performance(m_ovf, xval = TRUE)  # RMSE 8099.18
h2o.performance(m_ovf, test)         # RMSE 7155.508
