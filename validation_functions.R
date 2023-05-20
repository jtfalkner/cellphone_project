 
#functions to use 

#this computes training and validation RMSE 
compute_RMSE <- function(train_data,val_data,model,response_name){
  
  #compute training and validation error
  #train error
  train_error = sqrt(mean((train_data[[response_name]] - predict(model, train_data))^2)) 
  #validation error
  val_error = sqrt(mean((val_data[[response_name]] - predict(model, val_data))^2)) 
  
  #return relevant info
  return(list(train_error = train_error , val_error = val_error  ))
}



#the cross validation approach uses the caret package, and the format and syntax
#is nearly the same for each model
cross_Validate_OLS <- function(formula, train_data, method,train.control,...){
  
  #train with caret
  trained.model <- caret::train(formula, data = train_data,
                         method = method, 
                         trControl = train.control,
                         ...
  )
  
  #return relevant info
  return(list(cvRMSE = trained.model$results$RMSE ))
}

#the cross validation approach uses the caret package, and the format and syntax
#is nearly the same for each model.
#we will run three: ridge, lasso, and elasticnet
cross_Validate_RidgeLasso <- function(formula, train_data, train.control,...){
  
  #grid over lambdas to compute
  grid <- 10^seq(6,-3,length=100)
  
  #first ridge        
  ridge.mod <- caret::train(formula, data = train_data,
                            method = "glmnet", 
                            trControl = train.control,
                            tuneGrid = expand.grid(alpha =0, lambda=grid)
  )
  #save the best lambda nd corresponding RMSE
  #print(ridge.mod)
  #ridge.mod$results[which.min(ridge.mod$results$RMSE),]
  bestlam.ridge.mod <- ridge.mod$bestTune$lambda
  ridge.cvRMSE <-min(ridge.mod$results$RMSE)
  
  #next lasso        
  lasso.mod <- caret::train(formula, data = train_data,
                            method = "glmnet", 
                            trControl = train.control,
                            tuneGrid = expand.grid(alpha =1, lambda=grid)
  )
  #save the best lambda nd corresponding RMSE
  #print(lasso.mod)
  #lasso.mod$results[which.min(lasso.mod$results$RMSE),]
  bestlam.lasso.mod <- lasso.mod$bestTune$lambda
  lasso.cvRMSE <-min(lasso.mod$results$RMSE)
  
  #elastic net
  lgrid <- 10^seq(6,-3,length=25)
  agrid <- seq(0,1,length=4)
  ela.mod <- caret::train(formula, data = train_data,
                          method = "glmnet", 
                          trControl = train.control,
                          tuneGrid = expand.grid(alpha =agrid, lambda=lgrid)
  )
  #save the best lambda nd corresponding RMSE
  #print(ela.mod)
  #ela.mod$results[which.min(ela.mod$results$RMSE),]
  bestlam.ela.mod <- ela.mod$bestTune$lambda
  bestalpha.ela.mod <- ela.mod$bestTune$alpha
  ela.mod.cvRMSE <-min(ela.mod$results$RMSE)
  
  #return relevant info
  return(list(ridge.cvRMSE = ridge.cvRMSE, lasso.cvRMSE = lasso.cvRMSE, ela.cvRMSE=ela.mod.cvRMSE,
              ridge.lambda=bestlam.ridge.mod, lasso.lambda=bestlam.lasso.mod,ela.lambda=bestlam.ela.mod, 
              ela.alpha=bestalpha.ela.mod ))
  
  
}




#returns train/validate/test split
train_val_test_split <- function(y,pTrain,pVal,pTest){
  
  #set seed outside function, dont want that hidden from users 
  
  # creating training data as fraction of the dataset 
  train_index <- caret::createDataPartition(y, p = pTrain, list=FALSE)
  
  #create another split for validation and test on remaining indices
  all_ind = 1:length(y)
  non_train_ind = setdiff(all_ind,train_index)
  
  # Split the remaining between validate and test
  tmp_index <- caret::createDataPartition(non_train_ind, p = pVal/(pVal+pTest), list=FALSE)
  #these indices are validation
  val_index <- non_train_ind[tmp_index]
  
  #remaining is test
  test_index <-setdiff(all_ind,union(train_index,val_index))
  
  return(list(train = train_index, val = val_index,test = test_index ))
  
  
}

