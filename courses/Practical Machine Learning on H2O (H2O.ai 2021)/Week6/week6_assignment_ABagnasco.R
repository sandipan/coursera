## week 6 assignment choice #1
# it does not take long to train you can run it from the code "as is"
#declare library and start h2o server w/default 
library(h2o)
h2o.init()

#import data from supplied link
data = h2o.importFile("http://coursera.h2o.ai/house_data.3487.csv", destination_frame = "data")

#split data
parts<- h2o.splitFrame(data, c(0.9), destination_frames=c("train","test"), seed=123)

#check data size
sapply(parts, nrow)

# assign slitted frames to train and test h2o frames
train<- parts[[1]]
test<- parts[[2]]

# assign input columns and target variable
y<- "price"
input_cols<- setdiff(names(train), y)

# create and train model 1 using 5-fold cross-validation
# note this model was obtained through grid search
model1 = h2o.gbm(x = input_cols,
                 y = "price",
                 training_frame = train,
                 nfolds = 5,
                 seed = 1,
                 learn_rate = 0.01,
                 max_depth = 5,
                 sample_rate = 0.8,
                 col_sample_rate = 0.5,
                 ntrees = 1200,
                 nbins = 50,
                 nbins_top_level = 256,
                 nbins_cats = 128,
                 min_rows = 20,
                 keep_cross_validation_predictions = TRUE)

# compute cross-validation performance
perf1 <- h2o.performance(model1, xval = TRUE)

# performance is RMSE = 125395
perf1@metrics$RMSE

# create and train model2 using 5-fold cross-validation
model2 = h2o.randomForest(x = input_cols,
                          y = "price",
                          training_frame = train,
                          nfolds = 5,
                          seed = 1,
                          keep_cross_validation_predictions = TRUE)

# compute cross-validation performance
perf2 <- h2o.performance(model2, xval = TRUE)

# performance is not as good a model1 167899 
perf2@metrics$RMSE

# create and train model2 using 5-fold cross-validation
model3 = h2o.glm(x = input_cols,
                 y = "price",
                 training_frame = train,
                 nfolds = 5,
                 seed = 1,
                 lambda_search = TRUE,
                 keep_cross_validation_predictions = TRUE)

# compute cross-validation performance
perf3 <- h2o.performance(model3, xval = TRUE)

# let's see xval performance (worst of the three models) 367396
perf3@metrics$RMSE

# create ensemble with three naive models
ensemble <- h2o.stackedEnsemble(x = input_cols,
                                y = "price",
                                training_frame = train,
                                #seed = 1,
                                base_models = c(model1, model2, model3))

# determine ensemble performance on test data 
perf_ensemble <- h2o.performance(ensemble, newdata = test)

# performance is the best than the 3 native models (both for xval and test, here showing test), for test 115747  
perf_ensemble@metrics$RMSE

# final ensemble performance is better than the three individual naive models and better than the exercise target (RMSE = 115747 < 123000)

# finally save all three models and the ensemble
h2o.saveModel(model1, path = "C:\\Users\\bagna\\Documents", force=TRUE)
h2o.saveModel(model2, path = "C:\\Users\\bagna\\Documents", force=TRUE)
h2o.saveModel(model3, path = "C:\\Users\\bagna\\Documents", force=TRUE)
h2o.saveModel(ensemble, path = "C:\\Users\\bagna\\Documents", force=TRUE)

# lastly shut down h2o in order to release memory
h2o.shutdown()





