import h2o
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random
# h2o.clusterStatus()

data_x,data_y = datasets.make_regression(n_samples=10000,n_features=50)
pd.DataFrame(data_x)

h2o.init()

x = h2o.H2OFrame(data_x,destination_frame = "data_x")
y = h2o.H2OFrame(data_y,destination_frame = "data_y")
x.summary()
y.summary()

# h2o_to_pddf = h2o_data.as_data_frame()
# h2o_to_pddf

h2o.get_frame('data_y')

data = x.cbind(y)
data.describe

train,test = data.split_frame(ratios=[0.7],destination_frames = ['train','test'],seed=420)

train_cols = train.columns
train_cols.remove('C110')
train_cols

#Overfitting
from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(ntrees=2000,max_depth=7,nfolds=5,
                                        seed=1234,
                                        )

model_gbm.train(x=train_cols, y='C110', training_frame=train)
perf_train = model_gbm.model_performance(train)
perf_test = model_gbm.model_performance(test)
pred = model_gbm.predict(test)

perf_train

perf_test #Performs poorly on unseen data

#Without overfitting 
#Reduced the number of trees and nfolds

model_gbm_v2 = H2OGradientBoostingEstimator(ntrees=1000,max_depth=5,learn_rate=0.01,nfolds=3,
                                        seed=1234,
                                        )
model_gbm_v2.train(x=train_cols, y='C110', training_frame=train)
perf_train_v2 = model_gbm_v2.model_performance(train)
perf_test_v2 = model_gbm_v2.model_performance(test)

perf_train_v2

perf_test_v2

