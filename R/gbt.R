# Fit Gradient Boosted Trees
gbt_fit <- function(x, y, w, max_depth, leaf_size, bagn_frac, coln_frac,
                     shrinkage, num_trees, scl_pos_w, rpkg, objective) {
  if (rpkg[1] == "gbm") {
    # Set objective function
    if (objective[1] == "reg:linear") {
      distribution <- "gaussian"
    } else if (objective[1] == "binary:logistic") {
      distribution <- "bernoulli"
      pos_idx <- y == 1
      w[pos_idx] <- w[pos_idx] * scl_pos_w
    } else {
      stop(paste("Invalid objective function:", objective[1]))
    }

    # Fit gbm on the development data
    model <- gbm::gbm.fit(x = x,
                          y = as.numeric(y),
                          w = w,
                          distribution = distribution,
                          interaction.depth = max_depth,
                          n.minobsinnode = leaf_size,
                          bag.fraction = bagn_frac,
                          shrinkage = shrinkage,
                          n.trees = num_trees,
                          keep.data = FALSE,
                          verbose = FALSE)
  } else if (rpkg[1] == "xgb") {

    # Fit xgboost on the development data
    dtrain <- xgboost::xgb.DMatrix(as.matrix(x), label = y, weight = w,
                                   missing = NaN)
    model <- xgboost::xgboost(data = dtrain,
                              eta = shrinkage,
                              max_depth = max_depth,
                              min_child_weight = leaf_size,
                              subsample = bagn_frac,
                              colsample_bytree = coln_frac,
                              nround = num_trees,
                              scale_pos_weight = scl_pos_w,
                              nthread = 1,
                              objective = objective[1],
                              verbose = 0)
  } else {
    stop(paste("Invalid 'rpkg':", rpkg[1]))
  }

  return(model)
}


# Predict with Gradient Boosted Trees
gbt_predict <- function(model, x, rpkg, type = "response") {
  if (rpkg[1] == "gbm") {
    func <- tryCatch(utils::getFromNamespace("predict", ns = "gbm"),
                     error = function(e) { NULL })
    if (is.null(func)) {
      func <- tryCatch(utils::getFromNamespace("predict.gbm", ns = "gbm"),
                       error = function(e) { NULL })
      if (is.null(func)) {
        stop("Predict function for 'gbm' not found.")
      } else {
        pred <- func(model, x, model$n.trees, type = type)
      }
    } else {
      pred <- func(model, x, model$n.trees, type = type)
    }
  } else if (rpkg[1] == "xgb") {
    pred <- xgboost::predict(model, as.matrix(x))
  } else {
    stop(paste("Invalid 'rpkg':", rpkg[1]))
  }
  return(pred)
}
