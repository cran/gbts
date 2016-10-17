#' Hyperparameter Search for Gradient Boosted Trees
#'
#' This package provides hyperparameter optimization for Gradient Boosted Trees
#' (GBT) on binary classification and regression problems. The current version
#' provides two optimization methods:
#' \itemize{
#' \item Bayesian optimization:
#' \enumerate{
#' \item A probabilistic model is built to capture the relationship between
#' hyperparameters and their predictive performance.
#' \item Select the most predictive hyperparameter values (as suggested by
#' the probabilistic model) to try in the next iteration.
#' \item Train a GBT with the selected hyperparameter settings and compute its
#' out-of-sample predictive performance.
#' \item Update the probabilistic model with the new performance measure. Go
#' back to step 2 and repeat.
#' }
#' \item Random search: hyperparameters are selected uniformly at random in each
#' iteration.
#' }
#' In both approaches, each iteration uses cross-validation (CV) to develop GBTs
#' with the selected hyperparameter values on the training datasets followed by
#' performance assessment on the validation datasets. For Bayesian optimization,
#' validation performance is used to update the model of the relationship betwen
#' hyperparameters and performance. The final result is a set of CV models
#' having the best average validation performance. It does not re-run a new GBT
#' with the best hyperparameter values on the full training data. Prediction is
#' computed as the average of the predictions from the CV models.
#'
#' @docType package
#' @name gbts
NULL
#' @param x a data.frame of predictors. If \code{rpkg} (described below) is
#' set to \code{"gbm"}, then \code{x} is allowed to have categorical predictors
#' represented as factors. Otherwise, all predictors in \code{x} must be numeric.
#' @param y a vector of response values.  For binary classification, \code{y}
#' must contain values of 0 and 1. There is no need to convert \code{y} to a
#' factor variable. For regression, \code{y} must contain at least two unique
#' values.
#' @param w an optional vector of observation weights.
#' @param nitr an integer of the number of iterations for the optimization.
#' @param nlhs an integer of the number of Latin Hypercube samples (each sample
#' is a combination of hyperparameter values) used to generate the initial
#' performance model. This is used for Bayesian optimization only. Random search
#' ignores this argument.
#' @param nprd an integer of the number of samples (each sample is a combination
#' of hyperparameter values) at which performance prediction is made, and the
#' best is selected to run the next iteration of GBT.
#' @param kfld an integer of the number of folds for cross-validation used at
#' each iteration.
#' @param nwrk an integer of the number of computing workers (CPU cores) to be
#' used. If \code{nwrk} is less than the number of available cores on the
#' machine, it uses all available cores.
#' @param srch a character indicating the search method such that
#' \code{srch="bayes"} uses Bayesian optimization (default), and
#' \code{srch="random"} uses random search.
#' @param rpkg a character indicating which package of GBT to use. Setting
#' \code{rpkg="gbm"} uses the \code{gbm} R package (default). Setting
#' \code{rpkg="xgb"} uses the \code{xgboost} R package. Note that with
#' \code{gbm}, predictors can be categorical represented as factors, as opposed
#' to \code{xgboost} which requires all predictors to be numeric.
#' @param pfmc a character of the performance metric to optimize.
#' For binary classification, \code{pfmc} accepts:
#' \itemize{
#' \item \code{"acc"}: accuracy.
#' \item \code{"dev"}: deviance.
#' \item \code{"ks"}: Kolmogorov-Smirnov (KS) statistic.
#' \item \code{"auc"}: area under the ROC curve. This is used in conjunction
#' with the \code{cdfx} and \code{cdfy} arguments (described below) which
#' specify the cumulative distributions for the x-axis and y-axis of the ROC
#' curve, respectively. The default ROC curve is given by true positive rate
#' (on the y-axis) vs. false positive rate (on the x-axis).
#' \item \code{"roc"}: this is used when a point on the ROC curve is used as the
#' performance metric, such as the true positive rate at a fixed false positive
#' rate. This is used in conjunction with the \code{cdfx}, \code{cdfy}, and
#' \code{cutoff} arguments which specify the cumulative distributions for the
#' x-axis and y-axis of the ROC curve, and the cutoff (value on the x-axis) at
#' which evaluation of the ROC curve is obtained as a performance metric. For
#' example, if the desired performance metric is the true positive rate at
#' the 5\% false positive rate, specify \code{pfmc="roc"}, \code{cdfx="fpr"},
#' \code{cdfy="tpr"}, and \code{cutoff=0.05}.
#' }
#' For regression, \code{pfmc} accepts:
#' \itemize{
#' \item \code{"mse"}: mean squared error.
#' \item \code{"mae"}: mean absolute error.
#' \item \code{"rsq"}: r-squared (coefficient of determination).
#' }
#' @param cdfx a character of the cumulative distribution for the x-axis.
#' Supported values are
#' \itemize{
#' \item \code{"fpr"}: false positive rate.
#' \item \code{"fnr"}: false negative rate.
#' \item \code{"rpp"}: rate of positive prediction.
#' }
#' @param cdfy a character of the cumulative distribution for the y-axis.
#' Supported values are
#' \itemize{
#' \item \code{"tpr"}: true positive rate.
#' \item \code{"tnr"}: true negative rate.
#' }
#' @param cutoff a value in [0, 1] used for binary classification. If
#' \code{pfmc="acc"}, instances with probabilities <= \code{cutoff} are
#' predicted as negative, and those with probabilities > \code{cutoff} are
#' predicted as positive. If \code{pfmc="roc"}, \code{cutoff} can be used in
#' conjunction with the \code{cdfx} and \code{cdfy} arguments (described above)
#' to specify the operating point. For example, if the desired performance
#' metric is the true positive rate at the 5\% false positive rate, specify
#' \code{pfmc="roc"}, \code{cdfx="fpr"}, \code{cdfy="tpr"}, and
#' \code{cutoff=0.05}.
#' @param max_depth_range a vector of the minimum and maximum values for:
#' maximum tree depth.
#' @param leaf_size_range a vector of the minimum and maximum values for:
#' leaf node size.
#' @param bagn_frac_range a vector of the minimum and maximum values for:
#' bag fraction.
#' @param coln_frac_range a vector of the minimum and maximum values for:
#' fraction of predictors to try for each split.
#' @param shrinkage_range a vector of the minimum and maximum values for:
#' shrinkage.
#' @param num_trees_range a vector of the minimum and maximum values for:
#' number of trees.
#' @param scl_pos_w_range a vector of the minimum and maximum values for:
#' scale of weights for positive cases.
#' @param print_progress a logical of whether optimization progress should be
#' printed to the console.
#' @return A list of information with the following components:
#' \itemize{
#' \item \code{best_perf}: a numeric value of the best average validation
#' performance.
#' \item \code{best_idx}: an integer of the iteration index for the best
#' average validation performance.
#' \item \code{best_model_cv}: a list of cross-validation models with the
#' best average validation performance.
#' \item \code{perf_val_cv}: a matrix of cross-validation performances where
#' the rows correspond to iterations and the columns correspond to CV runs.
#' \item \code{params}: a data.frame of hyperparameter values visited during
#' the search. Each row of the data.frame comes from an iteration.
#' \item \code{total_time}: a numeric value of the total time used in minutes.
#' \item \code{objective}: a character of the objective function used.
#' \item \code{...}: the rest of the output are echo of the input arguments
#' (except for \code{x}, \code{y}, and \code{w}). See input argument
#' documentation for details.
#' }
#'
#' @seealso \code{\link{predict.gbts}},
#'          \code{\link{comperf}}
#'
#' @author Waley W. J. Liang <\email{wliang10@gmail.com}>
#'
#' @examples
#' \dontrun{
#' # Binary classification
#'
#' # Load German credit data
#' data(german_credit)
#' train <- german_credit$train
#' test <- german_credit$test
#' target_idx <- german_credit$target_idx
#' pred_idx <- german_credit$pred_idx
#'
#' # Train a GBT model with optimization on AUC
#' model <- gbts(train[, pred_idx], train[, target_idx], nitr = 200, pfmc = "auc")
#'
#' # Predict on test data
#' prob_test <- predict(model, test[, pred_idx])
#'
#' # Compute AUC on test data
#' comperf(test[, target_idx], prob_test, pfmc = "auc")
#'
#'
#' # Regression
#'
#' # Load Boston housing data
#' data(boston_housing)
#' train <- boston_housing$train
#' test <- boston_housing$test
#' target_idx <- boston_housing$target_idx
#' pred_idx <- boston_housing$pred_idx
#'
#' # Train a GBT model with optimization on MSE
#' model <- gbts(train[, pred_idx], train[, target_idx], nitr = 200, pfmc = "mse")
#'
#' # Predict on test data
#' prob_test <- predict(model, test[, pred_idx])
#'
#' # Compute MSE on test data
#' comperf(test[, target_idx], prob_test, pfmc = "mse")
#'}
#' @export
#' @import stats
#' @import earth
#' @importFrom doRNG %dorng%
#' @importFrom foreach %dopar%
gbts <- function(x, y,
                 w = rep(1, nrow(x)),
                 nitr = 100,
                 nlhs = floor(nitr / 2),
                 nprd = 1000,
                 kfld = 10,
                 nwrk = 2,
                 srch = c("bayes", "random"),
                 rpkg = c("gbm", "xgb"),
                 pfmc = c("acc", "dev", "ks", "auc", "roc", "mse", "rsq", "mae"),
                 cdfx = "fpr",
                 cdfy = "tpr",
                 cutoff = 0.5,
                 max_depth_range = c(2, 10),
                 leaf_size_range = c(10, 200),
                 bagn_frac_range = c(0.1, 1),
                 coln_frac_range = c(0.1, 1),
                 shrinkage_range = c(0.01, 0.1),
                 num_trees_range = c(50, 1000),
                 scl_pos_w_range = c(1, 10),
                 print_progress = TRUE) {
  # Check inputs
  if (missing(x) || !is.data.frame(x)) { stop("'x' is missing or not a data.frame.") }
  if (missing(y) || !is.vector(y)) { stop("'y' is missing or not a vector.") }
  if (length(y) != nrow(x)) { stop("length(y) != nrow(x).") }
  if (length(y) < 2) { stop("Not enough observations.") }
  if (length(which(w <= 0)) > 0) { stop("'w' has value(s) <= 0.") }
  if (nitr < 1 || !isint(nitr)) { stop("'nitr' < 1 or is not an integer.") }
  if (nlhs < 1 || !isint(nlhs)) { stop("'nlhs' < 1 or is not an integer.") }
  if (nprd < 1 || !isint(nprd)) { stop("'nprd' < 1 or is not an integer.") }
  if (kfld < 1 || !isint(kfld)) { stop("'kfld' < 1 or is not an integer.") }
  if (nwrk < 1 || !isint(nwrk)) { stop("'nwrk' < 1 or is not an integer.") }
  if (!(srch[1] %in% c("bayes", "active", "random"))) { stop("Invalid 'srch'.") }
  if (!(rpkg[1] %in% c("gbm", "xgb"))) { stop("Invalid 'rpkg'.") }
  if (rpkg[1] == "xgb" && length(which(unlist(lapply(x, is.factor)))) > 0) {
    stop("Cannot use 'xgb' because 'x' has factor(s).")
  }

  if (!(is.vector(max_depth_range) && length(max_depth_range) == 2 &&
        isint(max_depth_range[1]) && max_depth_range[1] >= 1 &&
        isint(max_depth_range[2]) && max_depth_range[2] > max_depth_range[1])) {
    stop("Invalid 'max_depth_range'.")
  }

  if (!(is.vector(leaf_size_range) && length(leaf_size_range) == 2 &&
        isint(leaf_size_range[1]) && leaf_size_range[1] >= 1 &&
        isint(leaf_size_range[2]) && leaf_size_range[2] > leaf_size_range[1] &&
        leaf_size_range[2] <= length(y))) {
    stop("Invalid 'leaf_size_range'.")
  }

  if (!(is.vector(bagn_frac_range) && length(bagn_frac_range) == 2 &&
        is.numeric(bagn_frac_range[1]) && bagn_frac_range[1] > 0 &&
        is.numeric(bagn_frac_range[2]) &&
        bagn_frac_range[2] > bagn_frac_range[1] && bagn_frac_range[2] <= 1)) {
    stop("Invalid 'bagn_frac_range'.")
  }

  if (!(is.vector(coln_frac_range) && length(coln_frac_range) == 2 &&
        is.numeric(coln_frac_range[1]) && coln_frac_range[1] > 0 &&
        is.numeric(bagn_frac_range[2]) &&
        coln_frac_range[2] > coln_frac_range[1] && coln_frac_range[2] <= 1)) {
    stop("Invalid 'coln_frac_range'.")
  }

  if (!(is.vector(shrinkage_range) && length(shrinkage_range) == 2 &&
        is.numeric(shrinkage_range[1]) && shrinkage_range[1] > 0 &&
        is.numeric(shrinkage_range[2]) &&
        shrinkage_range[2] > shrinkage_range[1] && shrinkage_range[2] <= 1)) {
    stop("Invalid 'shrinkage_range'.")
  }

  if (!(is.vector(num_trees_range) && length(num_trees_range) == 2 &&
        isint(num_trees_range[1]) && num_trees_range[1] >= 1 &&
        isint(num_trees_range[2]) && num_trees_range[2] > num_trees_range[1])) {
    stop("Invalid 'num_trees_range'.")
  }

  if (!(is.vector(scl_pos_w_range) && length(scl_pos_w_range) == 2 &&
        is.numeric(scl_pos_w_range[1]) && scl_pos_w_range[1] > 0 &&
        is.numeric(scl_pos_w_range[2]) &&
        scl_pos_w_range[2] > scl_pos_w_range[1])) {
    stop("Invalid 'scl_pos_w_range'.")
  }

  y <- as.numeric(y)
  if (pfmc[1] %in% c("acc", "dev", "ks", "auc", "roc")) {
    if (!identical(sort(unique(y)), c(0, 1))) {
      stop("Response variable 'y' has values other than 0 and 1.")
    }

    # Check cdfy and cdfx
    if (pfmc[1] == "auc" || pfmc[1] == "roc") {
      if (!(cdfy %in% c("tpr", "tnr"))) { stop("Invalid 'cdfy'.") }
      if (!(cdfx %in% c("fpr", "fnr", "rpp"))) { stop("Invalid 'cdfx'.") }
    }

    # Check operating point
    if (pfmc[1] == "roc" && !missing(cutoff) && !is.null(cutoff)) {
      if (length(cutoff) > 1) { stop("'cutoff' has length > 1.") }
      if (cutoff < 0 || cutoff > 1) { stop("Invalid 'cutoff'.") }
    }
    objective <- "binary:logistic"
  } else if (pfmc[1] %in% c("mse", "rsq", "mae")) {
    if (length(unique(y)) == 1) {
      stop("Response variable 'y' has only 1 unique value.")
    }
    objective <- "reg:linear"
  } else {
    stop("Invalid 'pfmc'.")
  }

  # Find out which hyperparameter(s) to search
  if (rpkg[1] == "gbm") {
    # Fix coln_frac_range to 1 for gbm
    coln_frac_range <- 1

    # Adjust maximum leaf size according to minimum bagging fraction
    # (TODO: adjust this individually in each iteration)
    train_size <- floor((nrow(x) / kfld)) * (kfld - 1)
    max_leaf_size <- floor((train_size * bagn_frac_range[1] - 2) / 2)
    if (length(leaf_size_range) == 1) {
      leaf_size_range <- min(leaf_size_range, max_leaf_size)
    } else {
      leaf_size_range[2] <- min(leaf_size_range[2], max_leaf_size)
    }
  }
  params_range <- rbind(max_depth_range, leaf_size_range, bagn_frac_range,
                        coln_frac_range, shrinkage_range, num_trees_range,
                        scl_pos_w_range)
  sp_idx <- as.vector(apply(params_range, 1, function(z) { z[1] != z[2] }))
  if (length(which(sp_idx)) == 0) { stop("All hyperparameters are fixed.") }

  # Propose hyperparameters
  if (srch[1] %in% c("bayes", "active")) {
    params <- propose_lhs_params(nlhs, max_depth_range,
                                 leaf_size_range, bagn_frac_range,
                                 coln_frac_range, shrinkage_range,
                                 num_trees_range, scl_pos_w_range)
  } else {
    params <- propose_random_params(nitr, max_depth_range,
                                    leaf_size_range, bagn_frac_range,
                                    coln_frac_range, shrinkage_range,
                                    num_trees_range, scl_pos_w_range)
  }

  # Set cross-validation (CV) partition indicies
  start_idx <- floor((nrow(x) / kfld) * 1:kfld - (nrow(x) / kfld) + 1)
  end_idx <- floor((nrow(x) / kfld) * 1:kfld)

  # Output variables
  perf_val_cv <- NULL
  best_perf <- 0
  best_idx <- NULL
  best_model_cv <- vector("list", kfld)
  ei <- NA

  # Parellel backend setup
  allcores <- parallel::detectCores() - foreach::getDoParWorkers()
  if (nwrk > allcores) {
    nwrk <- allcores
    warning("Number of available cores is less than 'nwrk'.")
  }
  cl <- parallel::makeCluster(nwrk)
  doParallel::registerDoParallel(cl)

  # Start hyperparameter search
  if (print_progress) { cat("Start hyperparameter search\n") }
  start_time <- Sys.time()
  set.seed(1)
  for (trial_idx in 1:nitr) {
    trial_time <- Sys.time()
    if (srch[1] %in% c("bayes", "active") && trial_idx > nlhs) {
      # Propose candidate hyperparameters around the current best setting
      Xs <- propose_local_params(nprd, params, best_idx,
                                 max_depth_range = max_depth_range,
                                 leaf_size_range = leaf_size_range,
                                 bagn_frac_range = bagn_frac_range,
                                 coln_frac_range = coln_frac_range,
                                 shrinkage_range = shrinkage_range,
                                 num_trees_range = num_trees_range,
                                 scl_pos_w_range = scl_pos_w_range)

      # Predict performance for the candidate hyperparameters
      perf_cnd <- matrix(NA, nrow = nprd, ncol = kfld)
      for (j in 1:kfld) { perf_cnd[, j] <- predict(perf_fits[[j]], Xs[, sp_idx]) }

      # Compute expected improvement
      if (pfmc[1] %in% c("dev", "mse", "mae")) {
        c_min <- best_perf
      } else if (pfmc[1] %in% c("acc", "ks", "auc", "roc", "rsq")) {
        perf_cnd[perf_cnd < 0] <- 0
        perf_cnd[perf_cnd > 1] <- 1
        perf_cnd <- log((1 - perf_cnd) / perf_cnd)
        c_min <- log((1 - best_perf) / best_perf)
      } else {
        stop("Invalid 'pfmc'.")
      }
      c_mean <- apply(perf_cnd, 1, mean)
      c_sd <- apply(perf_cnd, 1, sd)
      u <- (c_min - c_mean) / c_sd
      ei <- c_sd * (u * pnorm(u) + dnorm(u))

      # Select the candidate with the highest EI
      max_ei_idx <- which.max(ei)
      max_depth <- Xs$max_depth[max_ei_idx]
      leaf_size <- Xs$leaf_size[max_ei_idx]
      bagn_frac <- Xs$bagn_frac[max_ei_idx]
      coln_frac <- Xs$coln_frac[max_ei_idx]
      shrinkage <- Xs$shrinkage[max_ei_idx]
      num_trees <- Xs$num_trees[max_ei_idx]
      scl_pos_w <- Xs$scl_pos_w[max_ei_idx]
    } else {
      # Select randomly generated hyperparameters
      max_depth <- params$max_depth[trial_idx]
      leaf_size <- params$leaf_size[trial_idx]
      bagn_frac <- params$bagn_frac[trial_idx]
      coln_frac <- params$coln_frac[trial_idx]
      shrinkage <- params$shrinkage[trial_idx]
      num_trees <- params$num_trees[trial_idx]
      scl_pos_w <- params$scl_pos_w[trial_idx]
    }

    # Start CV in parallel
    mdpfcv <- foreach::foreach(j = 1:kfld, .options.RNG = 123) %dorng% {
      # Get validation indices
      val_idx <- start_idx[j]:end_idx[j]

      # Fit gbt on the development data
      model <- gbt_fit(x = x[-val_idx, ], y = y[-val_idx], w = w[-val_idx],
                       max_depth = max_depth, leaf_size = leaf_size,
                       bagn_frac = bagn_frac, coln_frac = coln_frac,
                       shrinkage = shrinkage, num_trees = num_trees,
                       scl_pos_w = scl_pos_w, rpkg = rpkg[1],
                       objective = objective)

      # Compute performance on the validation data
      pred <- gbt_predict(model, x[val_idx, ], rpkg[1], type = "response")
      perf <- comperf(y = y[val_idx], yhat = pred, w = w[val_idx],
                      pfmc = pfmc[1], cdfy = cdfy, cdfx = cdfx,
                      cutoff = cutoff)
      list(model = model, perf = perf)
    }

    # Collect validation performance
    perf_val_cv_i <- unlist(lapply(mdpfcv, function(z) { z$perf }))

    # Update validation performance
    perf_val_cv <- rbind(perf_val_cv, as.numeric(perf_val_cv_i))
    perf_val_mean <- apply(perf_val_cv, 1, mean)
    if (pfmc[1] %in% c("dev", "mse", "mae")) {
      best_perf <- min(perf_val_mean)[1]
      best_idx <- which.min(perf_val_mean)[1]
    } else if (pfmc[1] %in% c("acc", "ks", "auc", "roc", "rsq")) {
      best_perf <- max(perf_val_mean)[1]
      best_idx <- which.max(perf_val_mean)[1]
    }

    # Update best CV models
    if (best_idx == trial_idx) {
      for (j in 1:kfld) { best_model_cv[[j]] <- mdpfcv[[j]]$model }
    }

    # Save the selected hyperparameters
    if (srch[1] %in% c("bayes", "active") && trial_idx > nlhs) {
      params[trial_idx, ] <- c(max_depth, leaf_size, bagn_frac, coln_frac,
                               shrinkage, num_trees, scl_pos_w)
    }

    # Update modeling for performance
    if (srch[1] %in% c("bayes", "active") && trial_idx >= nlhs) {
      perf_fits <- foreach::foreach(j = 1:kfld, .options.RNG = 456) %dorng% {
        Y <- perf_val_cv[, j]
        perf_data <- data.frame(Y, params[, sp_idx])
        earth::earth(Y ~ ., data = perf_data, degree = ncol(perf_data), nk = 100)
      }
    }

    # Update progress
    trial_time <- as.numeric(difftime(Sys.time(), trial_time, units = "mins"))
    total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

    if (print_progress) {
      cat("i =", trial_idx, "| validation", paste0(pfmc[1], ":"),
          round(mean(perf_val_cv[trial_idx, ]), 6),
          "| best", paste0(pfmc[1], ":"), round(best_perf, 6),
          "| trial time:", paste0(round(trial_time, 2), "m"),
          "| total time:", paste0(round(total_time, 2), "m"), "\n")
    }
  }
  parallel::stopCluster(cl)

  output <- list(best_perf = best_perf,
                 best_idx = best_idx,
                 best_model_cv = best_model_cv,
                 perf_val_cv = perf_val_cv,
                 params = params,
                 total_time = total_time,
                 objective = objective[1],
                 nitr = nitr,
                 nlhs = nlhs,
                 nprd = nprd,
                 kfld = kfld,
                 nwrk = nwrk,
                 srch = srch[1],
                 rpkg = rpkg[1],
                 pfmc = pfmc[1],
                 max_depth_range = max_depth_range,
                 leaf_size_range = leaf_size_range,
                 bagn_frac_range = bagn_frac_range,
                 coln_frac_range = coln_frac_range,
                 shrinkage_range = shrinkage_range,
                 num_trees_range = num_trees_range,
                 scl_pos_w_range = scl_pos_w_range)
  class(output) <- "gbts"
  return(output)
}


#' Predict method for optimized Gradient Boosted Trees
#'
#' This function generates predictions for a given dataset using the returned
#' object from \code{\link{gbts}}. It first generates a set of predictions
#' for each cross-validation model, and then averaging them on the log-odds
#' scale (for binary classification) or on the response scale (for regression)
#' to produce the final prediction.
#'
#' @param object a model object returned from \code{\link{gbts}}.
#' @param x a data.frame of predictors. It has to follow the same format and
#' restrictions as the training dataset that generated the model.
#' @param nwrk an integer of the number of computing cores to be used. If
#' \code{nwrk} is less than the number of available cores on the machine, it
#' uses all available cores.
#' @param ... further arguments passed to or from other methods.
#' @return A numeric vector of predictions. In the case of binary classification,
#' predictions are probabilities.
#'
#' @seealso \code{\link{gbts}},
#'          \code{\link{comperf}}
#'
#' @author Waley W. J. Liang <\email{wliang10@gmail.com}>
#'
#' @export
#' @importFrom doRNG %dorng%
#' @method predict gbts
predict.gbts <- function(object, x, nwrk = 2, ...) {
  if (class(object) != "gbts") { stop("'object' is not of class 'gbts'") }

  # Parellel backend setup
  allcores <- parallel::detectCores() - foreach::getDoParWorkers()
  if (nwrk > allcores) {
    nwrk <- allcores
    warning("Number of available cores is less than 'nwrk'.")
  }
  cl <- parallel::makeCluster(nwrk)
  doParallel::registerDoParallel(cl)

  # Make predictions over the CV
  j <- 1
  pred_cv <- foreach::foreach(j = 1:object$kfld, .options.RNG = 123) %dorng% {
    gbt_predict(object$best_model_cv[[j]], x, object$rpkg, type = "response")
  }
  parallel::stopCluster(cl)
  pred_cv <- matrix(unlist(pred_cv), nrow = nrow(x), ncol = object$kfld)

  if (object$objective == "binary:logistic") {
    return(apply(pred_cv, 1, lgmean))
  } else {
    return(apply(pred_cv, 1, mean))
  }
}


# Propose hyperparameters uniformly at random
propose_random_params <-
  function(n, max_depth_range = NULL,
           leaf_size_range = NULL, bagn_frac_range = NULL,
           coln_frac_range = NULL, shrinkage_range = NULL,
           num_trees_range = NULL, scl_pos_w_range = NULL) {
  params <- list()

  # max_depth
  if (length(max_depth_range) == 1) {
    params[["max_depth"]] <- rep(max_depth_range, n)
  } else {
    params[["max_depth"]] <- floor(runif(n, max_depth_range[1], max_depth_range[2]))
  }

  # leaf_size
  if (length(leaf_size_range) == 1) {
    params[["leaf_size"]] <- rep(leaf_size_range, n)
  } else {
    params[["leaf_size"]] <- floor(runif(n, leaf_size_range[1], leaf_size_range[2]))
  }

  # bagn_frac
  if (length(bagn_frac_range) == 1) {
    params[["bagn_frac"]] <- rep(bagn_frac_range, n)
  } else {
    params[["bagn_frac"]] <- runif(n, bagn_frac_range[1], bagn_frac_range[2])
  }

  # coln_frac
  if (length(coln_frac_range) == 1) {
    params[["coln_frac"]] <- rep(coln_frac_range, n)
  } else {
    params[["coln_frac"]] <- runif(n, coln_frac_range[1], coln_frac_range[2])
  }

  # shrinkage
  if (length(shrinkage_range) == 1) {
    params[["shrinkage"]] <- rep(shrinkage_range, n)
  } else {
    params[["shrinkage"]] <- runif(n, shrinkage_range[1], shrinkage_range[2])
  }

  # num_trees
  if (length(num_trees_range) == 1) {
    params[["num_trees"]] <- rep(num_trees_range, n)
  } else {
    params[["num_trees"]] <- floor(runif(n, num_trees_range[1], num_trees_range[2]))
  }

  # scl_pos_w
  if (length(scl_pos_w_range) == 1) {
    params[["scl_pos_w"]] <- rep(scl_pos_w_range, n)
  } else {
    params[["scl_pos_w"]] <- runif(n, scl_pos_w_range[1], scl_pos_w_range[2])
  }

  return(data.frame(params))
}


# Propose hyperparameters using Latin Hypercube Sampling
propose_lhs_params <-
  function(nlhs, max_depth_range = NULL,
           leaf_size_range = NULL, bagn_frac_range = NULL,
           coln_frac_range = NULL, shrinkage_range = NULL,
           num_trees_range = NULL, scl_pos_w_range = NULL) {
  # Find out which hyperparameter(s) to search
  params_range <- rbind(max_depth_range, leaf_size_range, bagn_frac_range,
                        coln_frac_range, shrinkage_range, num_trees_range,
                        scl_pos_w_range)
  sp_idx <- as.vector(apply(params_range, 1, function(z) { z[1] != z[2] }))
  if (length(which(sp_idx)) == 0) { stop("All hyperparameters are fixed.") }
  nfix <- length(which(!sp_idx))

  # Generate initial hyperparameters using Latin Hypercube sampling
  params <- matrix(NA, nrow = nlhs, ncol = 7)
  #params[, sp_idx] <- tgp::lhs(nlhs, params_range[sp_idx, ])
  rang <- as.numeric(apply(params_range[sp_idx, ], 1, diff))
  minv <- as.numeric(apply(params_range[sp_idx, ], 1, min))
  params[, sp_idx] <- t(t(randomLHS(n = nlhs, k = length(which(sp_idx)))) *
                          rang + minv)
  params[, !sp_idx] <- matrix(rep(params_range[!sp_idx, 1], each = nlhs),
                              nrow = nlhs, ncol = nfix)
  params <- as.data.frame(params)
  names(params) <- c("max_depth", "leaf_size", "bagn_frac", "coln_frac",
                     "shrinkage", "num_trees", "scl_pos_w")
  params$max_depth <- floor(params$max_depth)
  params$leaf_size <- floor(params$leaf_size)
  params$num_trees <- floor(params$num_trees)
  return(params)
}


# Propose hyperparameters around the current best setting
propose_local_params <-
  function(nprd, params, best_idx, max_depth_range = NULL,
           leaf_size_range = NULL, bagn_frac_range = NULL,
           coln_frac_range = NULL, shrinkage_range = NULL,
           num_trees_range = NULL, scl_pos_w_range = NULL) {
  prms <- list()

  # max_depth
  if (length(max_depth_range) == 1) {
    prms[["max_depth"]] <- rep(max_depth_range, nprd)
  } else {
    prms[["max_depth"]] <- floor(rtnorm(nprd, mean = params$max_depth[best_idx],
                                        sd = 10 / 3, lower = max_depth_range[1],
                                        upper = max_depth_range[2]))
  }

  # leaf_size
  if (length(leaf_size_range) == 1) {
    prms[["leaf_size"]] <- rep(leaf_size_range, nprd)
  } else {
    prms[["leaf_size"]] <- floor(rtnorm(nprd, mean = params$leaf_size[best_idx],
                                        sd = 10 / 3, lower = leaf_size_range[1],
                                        upper = leaf_size_range[2]))
  }

  # bagn_frac
  if (length(bagn_frac_range) == 1) {
    prms[["bagn_frac"]] <- rep(bagn_frac_range, nprd)
  } else {
    prms[["bagn_frac"]] <- rtnorm(nprd, mean = params$bagn_frac[best_idx],
                                  sd = 0.1 / 3, lower = bagn_frac_range[1],
                                  upper = bagn_frac_range[2])
  }

  # coln_frac
  if (length(coln_frac_range) == 1) {
    prms[["coln_frac"]] <- rep(coln_frac_range, nprd)
  } else {
    prms[["coln_frac"]] <- rtnorm(nprd, mean = params$coln_frac[best_idx],
                                  sd = 0.1 / 3, lower = coln_frac_range[1],
                                  upper = coln_frac_range[2])
  }

  # shrinkage
  if (length(shrinkage_range) == 1) {
    prms[["shrinkage"]] <- rep(shrinkage_range, nprd)
  } else {
    prms[["shrinkage"]] <- rtnorm(nprd, mean = params$shrinkage[best_idx],
                                  sd = 0.01 / 3, lower = shrinkage_range[1],
                                  upper = shrinkage_range[2])
  }

  # num_trees
  if (length(num_trees_range) == 1) {
    prms[["num_trees"]] <- rep(num_trees_range, nprd)
  } else {
    prms[["num_trees"]] <- floor(rtnorm(nprd, mean = params$num_trees[best_idx],
                                        sd = 50 / 3, lower = num_trees_range[1],
                                        upper = num_trees_range[2]))
  }

  # scl_pos_w
  if (length(scl_pos_w_range) == 1) {
    prms[["scl_pos_w"]] <- rep(scl_pos_w_range, nprd)
  } else {
    prms[["scl_pos_w"]] <- rtnorm(nprd, mean = params$scl_pos_w[best_idx],
                                  sd = 1 / 3, lower = scl_pos_w_range[1],
                                  upper = scl_pos_w_range[2])
  }

  return(data.frame(prms))
}
