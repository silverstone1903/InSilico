#setwd("C:/Users/lenovo/Desktop/in silico/")

train_df <- read.csv("13321_2011_349_MOESM1_ESM_train.csv", T, ",")
test_df <- read.csv("13321_2011_349_MOESM2_ESM_test.csv", T, ",")


# library(doParallel)
# cl <- makeCluster(detectCores())
# registerDoParallel(cl)
# getDoParWorkers()


head(train_df)
tail(train_df)
head(test_df)
tail(test_df)



#### CART ####

library(rpart)
library(rpart.plot)
set.seed(1903)
dtree <- rpart(Activity ~ ., data = train_df, method = "class", 
               control = rpart.control(xval = 10, maxdepth = 8))

printcp(dtree)
summary(dtree)
plotcp(dtree)
rsq.rpart(dtree)


plot(dtree, uniform=TRUE, 
     main="Classification Tree")
text(dtree, use.n=TRUE, all=TRUE, cex=.8)


pdtree <- prune(dtree, cp = dtree$cptable[which.min(dtree$cptable[,"xerror"]),"CP"])

# plot the pruned tree 
plot(pdtree, uniform=TRUE, 
     main="Pruned Classification Tree")
text(pdtree, use.n=TRUE, all=TRUE, cex=.8)


predict_dt <- as.data.frame(predict(dtree, test_df))
head(predict_dt)

predict_pdt <- as.data.frame(predict(pdtree, test_df))
head(predict_pdt)

#prediction for tree
predict_dt$Act <- NA
for (i in 1:nrow(predict_dt))
{ 
predict_dt[i,3] <- ifelse(predict_dt[i, 1] > 0.5, "mutagen", "nonmutagen")
}

(table(predict_dt$Act, test_df$Activity)[1] + table(predict_dt$Act, test_df$Activity)[4])/nrow(test_df) #0.71627256

#prediction for pruned tree
predict_pdt$Act <- NA
for (i in 1:nrow(predict_pdt))
{ 
  predict_pdt[i,3] <- ifelse(predict_pdt[i, 1] > 0.5, "mutagen", "nonmutagen")
}
table(predict_pdt$Act, test_df$Activity)

(table(predict_pdt$Act, test_df$Activity)[1] + table(predict_pdt$Act, test_df$Activity)[4])/nrow(test_df) #0.71627256


#tree viz
only_count <- function(x, labs, digits, varlen)
{
  paste(x$frame$n)
}

boxcols <- c("Red", "Blue")[pdtree$frame$yval]

par(xpd=TRUE)
prp(pdtree, faclen = 0, cex = 0.8, node.fun=only_count, box.col = boxcols)
legend("topright", legend = c("mutagen","nonmutagen"), fill = c("Red", "Blue"),
       title = "Activity", cex = 0.6, bty = "n")

gc(reset = T)





#### random forest ####

library(randomForest)
set.seed(1903)
system.time(
rf <- randomForest(Activity ~ ., data = train_df, importance = T, 
                   mtry = 12, proximity = T, oob.prox = T)) # default ntree = 500


print(rf)
print(importance(rf, type = 1))
varImpPlot(rf, n.var = 20)

predict_rf <- predict(rf, test_df)
head(predict_rf)
table(predict_rf, test_df$Activity)
print((table(predict_rf, test_df$Activity)[1] + table(predict_rf, test_df$Activity)[4])/nrow(test_df)) #0.8468787


gc(reset = T)

#### gbm ####
library(gbm)
train_gbm <- train_df
test_gbm <- test_df

# Num. transformation of target variable on train set
train_gbm$Activity <- as.numeric(train_gbm$Activity)
train_gbm$Activity[train_gbm$Activity == 2] <- 0

tail(train_gbm$Activity)
head(train_gbm$Activity)

# Num. transformation of target variable on test set
test_gbm$Activity <- as.numeric(test_gbm$Activity)
test_gbm$Activity[test_gbm$Activity == 2] <- 0

tail(test_gbm$Activity)
head(test_gbm$Activity)
head(train_gbm)

#gb model
set.seed(1903)
system.time(
gbmodel <- gbm(formula = Activity ~., data = train_gbm, interaction.depth = 5, shrinkage = 0.05,
               distribution = "bernoulli", n.trees = 500, cv.folds = 10, 
               verbose = T, n.cores = 4)
)
summary(gbmodel)
# find the best iteration
best.iter = gbm.perf(gbmodel)


predict_gbm <- predict.gbm(gbmodel, newdata = test_gbm, 
                           n.trees = gbm.perf(gbmodel, plot.it = F), type = "response")
predict_gbm <- round(predict_gbm)

head(predict_gbm, 10)
table(predict_gbm, test_gbm$Activity)

(table(predict_gbm, test_gbm$Activity)[1] + table(predict_gbm, test_gbm$Activity)[4])/nrow(test_gbm) #0.7791519


gc(reset = T)

#### xgboost ####
library(xgboost)
library(Matrix)

train_xgb <- read.csv("13321_2011_349_MOESM1_ESM_train.csv", T, ",")
test_xgb <- read.csv("13321_2011_349_MOESM2_ESM_test.csv", T, ",")

train_label <- factor(train_xgb$Activity, levels = c("mutagen", "nonmutagen"),
                            labels = c(1,0))
test_label <- factor(test_xgb$Activity, levels = c("mutagen", "nonmutagen"),
                           labels = c(1,0))

train_label <- as.integer(as.character(train_label))
test_label <- as.integer(as.character(test_label))



data <- rbind(train_xgb, test_xgb)
head(data)


#data_sparse <- sparse.model.matrix(Activity~., data = data)

train_sparse <- sparse.model.matrix(Activity~., data = train_xgb)
test_sparse <- sparse.model.matrix(Activity~., data = test_xgb)


dtrain <- xgb.DMatrix(data = train_sparse, label = train_label)
dtest <- xgb.DMatrix(data = test_sparse, label = test_label) 

set.seed(1903)
temp_model <- xgb.cv(data = dtrain, label = train_label,
                     nfold = 10,
                     nrounds = 500,
                     max_depth = 6,
                     eta = 0.05,
                     subsample = 0.7,
                     colsample_bytree = 0.7,
                     metrics = "error",
                     maximize = FALSE,
                     early_stopping_rounds = 10,
                     min_child_weight = 1,
                     objective = "binary:logistic",
                     print_every_n = 10,
                     verbose = TRUE)

best_it <- temp_model$best_iteration

set.seed(1903)
system.time(
  xgb_model <- xgb.train(data = dtrain,
                          nrounds = best_it,
                          early_stopping_rounds = 10,
                          max_depth = 6,
                          eta = 0.05, 
                          subsample = 0.7,
                          min_child_weight = 1,
                          colsample_bytree = 0.7,
                          eval_metric = "error",
                          maximize = FALSE,
                          objective = "binary:logistic",
                          print_every_n = 10,
                          verbose = TRUE,
                          watchlist = list(train = dtrain)))


predicted_xgb <- predict(xgb_model, dtest)
head(predicted_xgb)

predicted_xgb2 <- predict(xgb_model, dtest, ntreelimit = best_it)
head(predicted_xgb2)

prediction <- as.numeric(predicted_xgb2 > 0.5)
print(head(prediction))

err <- mean(as.numeric(predicted_xgb2 > 0.5) != test_label)
print(paste(" test-error = ", err))

table(prediction, test_label)
(table(prediction, test_label)[1] + table(prediction, test_label)[4])/nrow(test_xgb) # 0.8421673


importance <- xgb.importance(feature_names = data_sparse@Dimnames[[2]], 
                             model = xgb_model)

xgb.ggplot.deepness(xgb_model)
xgb.ggplot.importance(top_n = 10, importance_matrix = importance)
xgb.model.dt.tree(feature_names = train_sparse@Dimnames[[2]], model = xgb_model, 
                  n_first_tree = 2)

importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)
xgb.dump(xgb_model, with_stats = T)
xgb.plot.tree(model = xgb_model, n_first_tree = 2, feature_names = train_sparse@Dimnames[[2]])
xgb.plot.multi.trees(model = temp_model, feature_names = train_sparse@Dimnames[[2]], features_keep = 3 )


#### confusion matrix viz ####
caret::confusionMatrix(predict_pdt$Act, test_df$Activity)
caret::confusionMatrix(predict_rf, test_df$Activity)
caret::confusionMatrix(predict_gbm, test_gbm$Activity)
caret::confusionMatrix(prediction, test_label)

#fourfoldplot
par(mfrow = c(2,2))
fourfoldplot(caret::confusionMatrix(predict_pdt$Act, test_df$Activity)$table, main = "CART")
fourfoldplot(caret::confusionMatrix(predict_rf, test_df$Activity)$table, main = "Random Forest")
fourfoldplot(caret::confusionMatrix(prediction, test_label)$table, main = "GBM")
fourfoldplot(caret::confusionMatrix(predict_gbm, test_gbm$Activity)$table, main = "xgboost")
