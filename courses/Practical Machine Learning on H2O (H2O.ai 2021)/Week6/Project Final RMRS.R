#######################################################################  Presentation  ###############################################################################
# Rosa Romero Soler
# 23/07/2021

# Steps

# STEP 1 is to start h2o, load your chosen data set(s) and follow the project-specific data manipulation steps.
# At the end of this step you will have `train`, `test`, `x` and `y` variables, and possibly `valid` also. 
# Check you have the correct number of rows and columns (as specified in the project description) before moving on.

# STEP 2 is to train at least four different models, using at least three different supervised algorithms. 
# Save all your models. You may use any data engineering you wish, but must not bring in any additional external data sources.

# STEP 3 is to train a stacked ensemble of the models you made in step two.
# Repeat steps two and three until your best model (which is usually your ensemble model, but does not have to be) has the minimum required performance on the validation data.
# Note: only one model has to achieve the minimum required performance. If multiple models achieve it, choose the best performing one to deliver.

# STEP 4 is to get the performance on the test data of your chosen model/ensemble, and confirm that this also reaches the minimum target on the test data. Record your model performance in comments at the end of your script.

# STEP 5 is to start a fresh session and confirm your script will run, reproduce the results, and that there are no bugs or typos.

# The final step is to submit your script, and then evaluate the scripts of your fellow students.

# The complete execution of the file on my laptop took 4' 23'' 

# Start        : 2021-07-26 08:42:54
# The end      : 2021-07-26 08:47:17

#######################################################################  STEP 1  ###############################################################################

cat("Start      :", as.character.Date(Sys.time()))


# Start h2o
library(h2o)
h2o.init()

# Load my chosen data set
url<- 'http://coursera.h2o.ai/house_data.3487.csv'
data<- h2o.importFile(url)

h2o.describe(data)

# Number of rows = 21.613
h2o.nrow(data)

# I split date into year and month columns. I combine them into a numeric date column

  # I build a dataframe with the 3 indicated columns, year, month and a numerical combination of both 
  # and then I add these columns to data 
    # It is necessary to verify by means of the describe of the columns that there are no missing values 
    # so that the numerical operations do not give problems

  d_time <- data.frame(
                        year=as.numeric(substr(as.vector(h2o.ascharacter(data$date)),start = 1, stop = 4)),
                        month=as.numeric(substr(as.vector(h2o.ascharacter(data$date)),start = 5, stop = 6)),
                        date_numeric= as.numeric(substr(as.vector(h2o.ascharacter(data$date)),start = 1, stop = 4))*
                          as.numeric(substr(as.vector(h2o.ascharacter(data$date)),start = 5, stop = 6))
  )  
  
  # I add the generated columns to the original data frame
  data<-h2o.cbind(data,as.h2o(d_time))
  
  # I am going to consider that the date column contains the date, year and month, of when the value of the house 
  # is taken, I imagine that through an appraisal. That is why I will use that year column as a reference in addition 
  # to the years of construction of the moment in which the home appraisal was carried out and also, as a reference 
  # to know how long the reform has been carried out, if at all.
  # Therefore I generate and add the following columns to the original dataframe:
    # years_builts: age of the home at the time of appraisal
    # years_renovated: age of the reform at the time of appraisal
    # is_renovated: indicates if the house has been reformed or not
  
      years_builts<-data$year-data$yr_built
      years_renovated<-data$year-data$yr_renovated
      is_renovated<-as.factor(data$yr_renovated>0)
      names(is_renovated)[1] = c("is_renovated")
      names(years_builts)[1] = c("years_builts")
      names(years_renovated)[1] = c("years_renovated")
  
  #  Add the following columns to the original dataframe   
  data<-h2o.cbind(data,years_builts,years_renovated,is_renovated)
    


#it is checked again that our h2o frame contains the new columns year, month , date_numeric, years_builts,years_renovated
# and is_renovated:
  
h2o.describe(data)

# Of the columns with information, we transform into a factor those whose values indicate it:

  data$waterfront   <- h2o.asfactor(data$waterfront)
  data$view         <- h2o.asfactor(data$view)
  data$condition    <- h2o.asfactor(data$condition)
  data$grade        <- h2o.asfactor(data$grade)
  data$zipcode      <- h2o.asfactor(data$zipcode)

#it is checked again that our h2o frame 
h2o.describe(data)


# Separate in training and test: 

parts <- h2o.splitFrame(data,0.9,seed = 123)
train <- parts[[1]]
test  <- parts[[2]]

# Verification of correct distribution of the data according to the statement

h2o.nrow(train) # 19.462
h2o.nrow(test)  #  2.151

# Target variable is indicated
y = "price"
# As predictor columns we are going to use all of them except id because it does not provide information, the date 
# column because we have decomposed them and I have used the year for later calculations that I do incorporate,
# and price because it is the target variable. Longitude and latitude, I assume they will be incorporated and correlated 
# with zipcode
x = setdiff(colnames(data),c(colnames(data)[1:3],"lat","long","year","month","date_numeric","yr_built","yr_renovated"))




#######################################################################  STEP 2  ###############################################################################

# The algorithms to be used for the three independent models are:

  # GLM 
  # GBM 
  # Random Forest 
  # Deep Learning

# Why did I select these models? Because I did an AutoML and the best models it returns are these.
# Why I have not used grid for the discovery of the best parameters? Because by testing the best AutoML 
# models and measuring performance, I was already hitting the RMSE target

# Cross validation is used for all 3 models and the same number of folders is set for all of them
folds_for_models = 10


# 
m_glm = h2o.glm(
                  x,
                  y,
                  training_frame = train,
                  family = "gamma", 
                  link = "log",
                  model_id = "glm",
                  nfolds =folds_for_models,
                  fold_assignment = "Modulo" ,
                  keep_cross_validation_predictions = TRUE,
                  seed = 123
               )

m_glm  
h2o.rmse(h2o.performance(m_glm,test))

m_gbm = h2o.gbm(
                  x,
                  y,
                  training_frame = train,
                  model_id = "gbm",
                  distribution = "gamma",
                  nfolds =folds_for_models,
                  fold_assignment = "Modulo" ,
                  keep_cross_validation_predictions = TRUE,
                  seed = 123
)


m_gbm # 132566.1
h2o.performance(m_gbm,test)
h2o.rmse(h2o.performance(m_gbm,test))


m_rf = h2o.randomForest(
                  x,
                  y,
                  training_frame = train,
                  model_id = "rf",
                  nfolds =folds_for_models,
                  fold_assignment = "Modulo" ,
                  keep_cross_validation_predictions = TRUE,
                  seed = 123
)

m_rf # 143481.8
h2o.performance(m_rf,test)
h2o.rmse(h2o.performance(m_rf,test))



m_dl= h2o.deeplearning(x,y,train,model_id="dl",
                       nfolds =folds_for_models,
                       fold_assignment = "Modulo" ,
                       keep_cross_validation_predictions = TRUE,
                       seed = 123)

m_dl
h2o.performance(m_dl,test)
h2o.rmse(h2o.performance(m_dl,test))


#######################################################################  STEP 3  ###############################################################################


model_ids=list(m_glm@model_id,m_gbm@model_id,m_rf@model_id,m_dl@model_id )


m_SE <-  h2o.stackedEnsemble(
                                x,
                                y,
                                training_frame = train,
                                model_id="SE_glm_gbm_rf_dl",
                                base_models = model_ids#,
                                #seed=123
)

m_SE
h2o.performance(m_SE,test)
h2o.rmse(h2o.performance(m_SE,test))

#######################################################################  STEP 4  ###############################################################################


models <- c(m_glm,m_gbm,m_rf,m_dl,m_SE)

sapply(models,h2o.rmse)
sapply(models,h2o.rmse,xval=TRUE)

perfs <- lapply(models, h2o.performance,test)

Models=c(model_ids,"SE")
Performance.test=as.vector(sapply(perfs, h2o.rmse))


# Performance of the models in the test set
Performance_models<-data.frame(Models,Performance.test)

# My Performance result: 
Performance_models[5,2] # 118715.3


#######################################################################  Final STEP   ###############################################################################

# I save the models
h2o.saveModel(m_glm,"./Models/",force = TRUE)
h2o.saveModel(m_gbm,"./Models/",force = TRUE)
h2o.saveModel(m_rf,"./Models/",force = TRUE)
h2o.saveModel(m_dl,"./Models/",force = TRUE)
h2o.saveModel(m_SE,"./Models/",force = TRUE)

cat("The end      :", as.character.Date(Sys.time()))







