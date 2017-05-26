### h2o library ###
library(h2o)
h2o.init(nthreads = -1)
h2o.removeAll()
#h2o.shutdown(prompt=FALSE)


#setwd("C:/Users/lenovo/Desktop/in silico/")

train_df <- read.csv("13321_2011_349_MOESM1_ESM_train.csv", T, ",")
test_df <- read.csv("13321_2011_349_MOESM2_ESM_test.csv", T, ",")

test.hex <- as.h2o(test_df)
data.hex <- as.h2o(train_df)

splits <- h2o.splitFrame(
  data = data.hex, 
  ratios = c(0.8),  
  destination_frames = c("train.hex", "valid.hex"), seed = 2017
)
train <- splits[[1]]
valid <- splits[[2]]
target <- "Activity"
predictors <- setdiff(colnames(train), c("Activity"))
                       


system.time(
model <- h2o.deeplearning(x = predictors, 
                          y = target,
                          rate = 0.01,
                          stopping_rounds = 10,
                          stopping_metric = "misclassification",
                          distribution = "AUTO",
                          nfolds = 10,
                          training_frame = train, validation_frame = valid, 
                          activation = "Tanh", 
                          hidden = c(80,50), epochs = 100,
                          variable_importances = T
                          )
)

print(model@model$model_summary)
performance <- h2o.performance(model = model, valid = T)
print(performance)
predictions <- predict(object = model, test.hex)
predictions.R <- as.data.frame(predictions)
head(predictions.R)

plot(model)
plot(h2o.performance(model))
plot(h2o.performance(model, valid = T))

head(as.data.frame(h2o.varimp(model)))
feat_var <- as.data.frame(h2o.varimp(model))

#h2o.deepfeatures(model, train, layer = 1)

#h2o.hit_ratio_table(model,valid = T)[1,2]

### grid search for dl model ###

hyper_params <- list(
  hidden=list(c(32,32,32),c(64,64)),
  input_dropout_ratio=c(0,0.05),
  rate=c(0.01,0.05),
  rate_annealing=c(1e-8,1e-7,1e-6)
)

hyper_params

grid <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid", 
  training_frame = train,
  validation_frame = valid, 
  x = predictors, 
  y = target,
  epochs = 50,
  stopping_metric = "misclassification",
  stopping_tolerance = 1e-2,        # stop when misclassification does not improve by >=1% for 2 scoring events
  stopping_rounds = 2,
  score_validation_samples = 10000, # downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ##don't score more than 2.5% of the wall time
  adaptive_rate = F,                # manually tuned learning rate
  momentum_start = 0.5,  ## manually tuned momentum
  momentum_stable = 0.9, 
  momentum_ramp = 1e7, 
  l1=1e-5,
  l2=1e-5,
  activation = c("Rectifier"),
  hyper_params=hyper_params
)
grid


grid <- h2o.getGrid("dl_grid", sort_by="err",
                    decreasing=FALSE)
grid


h2o.getGrid("dl_grid", sort_by = "logloss", decreasing = FALSE)

## Find the best model and its full set of parameters
grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]])
best_model

print(best_model@allparameters)
print(h2o.performance(best_model, valid = T))
print(h2o.logloss(best_model, valid = T))

plot(best_model)
plot(h2o.performance(best_model))
plot(h2o.performance(best_model, valid = T))

### grid searh for deep learning end ###


### random search for deep learning ###

hyper_params <- list(
  activation=c("Rectifier","Tanh",
               "Maxout","RectifierWithDropout",
               "TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20, 20),c(50, 50),
              c(30, 30, 30),c(25, 25, 25, 25)),
  input_dropout_ratio=c(0, 0.05),
  l1=seq(0, 1e-4, 1e-6),
  l2=seq(0, 1e-4, 1e-6)
)
hyper_params

## Stop once the top 5 models are within 1% of each other (i.e., the windowed average varies less than 1%)
search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 720, max_models = 100, seed = 2017, 
  stopping_rounds = 5, stopping_tolerance = 1e-2)

dl_random_grid <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "dl_grid_random",
  training_frame = train,
  validation_frame = valid, 
  x = predictors, 
  y = target,
  epochs = 50,
  stopping_metric = "logloss",
  stopping_tolerance = 1e-2,        ## stop when logloss does not improve by >=1% for 2 scoring events
  stopping_rounds = 2,
  score_validation_samples = 1000, ## downsample validation set for faster scoring
  score_duty_cycle = 0.025,         ## don't score more than 2.5% of the wall time
  max_w2 = 10,                      ## can help improve stability for Rectifier
  hyper_params = hyper_params,
  search_criteria = search_criteria
)                                
grid <- h2o.getGrid("dl_grid_random",sort_by = "logloss", decreasing = FALSE)
grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model


grid <- h2o.getGrid("dl_grid",sort_by="err",decreasing=FALSE)
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest classification error (on validation, since it was available during training)
h2o.confusionMatrix(best_model,valid=T)
best_params <- best_model@allparameters
best_params$activation
best_params$hidden
best_params$input_dropout_ratio
best_params$l1
best_params$l2

plot(best_model)
plot(h2o.performance(best_model))
plot(h2o.performance(best_model, valid = T))
plot(h2o.performance(best_model, newdata = test.hex))

### random search for deep learning end ###


### save model ###
#path <- h2o.saveModel(model, 
#                     path="C:/Users/User/Desktop/Kaggle_enis/in silico/", force=TRUE)

print(path)

#m_loaded <- h2o.loadModel(path)
#summary(m_loaded)


# random forest
system.time(
  rf <- h2o.randomForest(y = target, 
                         x = predictors, 
                         training_frame = train,
                         validation_frame = valid,
                         ntrees = 1000, 
                         mtries = 12, 
                         max_depth = 8,
                         seed = 1903, 
                         sample_rate = 0.8, 
                         nfolds = 10, 
                         stopping_metric = "logloss",
                         col_sample_rate_per_tree = 0.8,
                         stopping_rounds = 25))


h2o.auc(h2o.performance(rf, valid = T))
h2o.performance(rf)
rfvimp <- h2o.varimp(rf)
system.time(predict.rforest <- h2o.predict(rf, test.hex))
head(predict.rforest)

# random forest end

### gradient boosting ###
system.time(
  gbm <- h2o.gbm(y = target, x = predictors,
              distribution = "bernoulli",
              training_frame = train,
              validation_frame = valid,
              stopping_metric = "logloss",
              ntrees=1000,
              max_depth=5,
              learn_rate=0.01,
              stopping_rounds = 15,
              stopping_tolerance = 1e-4,
              sample_rate = 0.7,   
              col_sample_rate = 0.7,                                                   
              seed = 1903,    
              score_tree_interval = 10,
              nfolds = 10))

h2o.mse(h2o.performance(gbm, valid = TRUE))

h2o.performance(gbm)
h2o.performance(gbm, valid = T)
pred  <- h2o.predict(gbm,test.hex)
head(pred)

plot(gbm)
plot(h2o.performance(gbm))
plot(h2o.performance(gbm, valid = T))

head(as.data.frame(h2o.varimp(gbm)))

### gradient boosting end ###
