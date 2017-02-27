#' Hyperparameter Search for Gradient Boosted Trees
#'
#' This package implements hyperparameter optimization for Gradient Boosted
#' Trees (GBT) on binary classification and regression problems. The current
#' version provides two optimization methods:
#' \itemize{
#' \item Bayesian optimization:
#' \enumerate{
#' \item A predictive model is built to capture the relationship between GBT
#' hyperparameters and the resulting predictive performance.
#' \item Select the best hyperparameter setting (determined by a pre-specified
#' criterion) to try in the next iteration.
#' \item Train GBT on the selected hyperparameter setting and compute validation
#' performance.
#' \item Update the predictive model with the new validation performance. Go
#' back to step 2 and repeat.
#' }
#' \item Random search: each GBT is built with a randomly selected
#' hyperparameter setting.
#' }
#' Instead of returning a single GBT in the final output, an ensemble of GBTs
#' is produced via the method of ensemble selection. It selects GBTs with
#' replacement from a library into the ensemble, and returns the ensemble with
#' best validation performance. Model library and validation performance are
#' obtained from the hyperparameter search described above, by building GBTs
#' with different hyperparameter settings on the training dataset and obtaining
#' their performances on the validation dataset, based on cross-validation (CV).
#' Since selection from the library is done with replacement, each GBT may be
#' selected more than once into the ensemble. This function returns an ensemble
#' that contains only the unique GBTs with model weights calculated as the
#' number of model duplicates divided by the ensemble size. Each unique GBT in
#' the ensemble is re-trained on the full training data. Prediction is computed
#' as the weighted average of predictions from the re-trained GBTs.
#'
#' @docType package
#' @name gbts
NULL
#' @param x a data.frame of predictors. Categorical predictors represented as
#' \code{factors} are accepted.
#' @param y a vector of response values. For binary classification, \code{y}
#' must contain values of 0 and 1. It is unnecessary to convert \code{y} to a
#' factor variable. For regression, \code{y} must contain at least two unique
#' values.
#' @param w an optional vector of observation weights.
#' @param nitr an integer of the number of hyperparameter settings that are
#' sampled. For Bayesian optimization, \code{nitr} must be larger than
#' \code{nlhs}.
#' @param nlhs an integer of the number of Latin Hypercube samples (each sample
#' is a hyperparameter setting) used to initialize the predictive model of GBT
#' performance. This is used for Bayesian optimization only. After
#' initialization, sequential search continues for \code{nitr-nlhs} iterations.
#' @param nprd an integer of the number of hyperparameter settings at which
#' GBT performance is estimated using the predictive model and the best is
#' selected to train GBT in the next iteration.
#' @param kfld an integer of the number of folds for cross-validation.
#' @param srch a character of the search method such that \code{srch="bayes"}
#' performs Bayesian optimization (default), and \code{srch="random"} performs
#' random search.
#' @param nbst an integer of the number of bootstrap samples to construct the
#' predictive model of GBT performance.
#' @param ensz an integer value of the ensemble size - number of GBTs selected
#' into the ensemble. Since ensemble selection is done with replacement, the
#' number of unique GBTs may be less than \code{ensz}, but the sum of model
#' weights always equals \code{ensz}.
#' @param nwrk an integer of the number of computing workers to use on a single
#' machine.
#' @param rpkg a character indicating which R package implementation of GBT to
#' use. Currently, only the \code{gbm} R package is supported.
#' @param pfmc a character of the performance metric used as the optimization
#' objective.
#' For binary classification, \code{pfmc} accepts:
#' \itemize{
#' \item \code{"acc"}: accuracy
#' \item \code{"dev"}: deviance
#' \item \code{"ks"}: Kolmogorov-Smirnov (KS) statistic
#' \item \code{"auc"}: area under the ROC curve. Use the \code{cdfx} and
#' \code{cdfy} arguments to specify the cumulative distributions for the x-axis
#' and y-axis of the ROC curve, respectively. The default ROC curve is given by
#' true positive rate (y-axis) vs. false positive rate (x-axis).
#' \item \code{"roc"}: rate on the y-axis of the ROC curve at a particular
#' decision point (threshold) on the x-axis specified by the \code{dspt}
#' argument. For example, if the desired performance metric is true positive
#' rate at the 5\% false positive rate, specify \code{pfmc="roc"},
#' \code{cdfx="fpr"}, \code{cdfy="tpr"}, and \code{dspt=0.05}.
#' }
#' For regression, \code{pfmc} accepts:
#' \itemize{
#' \item \code{"mse"}: mean squared error
#' \item \code{"mae"}: mean absolute error
#' \item \code{"rsq"}: r-squared (coefficient of determination)
#' }
#' @param cdfx a character of the cumulative distribution for the x-axis.
#' Supported values are
#' \itemize{
#' \item \code{"fpr"}: false positive rate
#' \item \code{"fnr"}: false negative rate
#' \item \code{"rpp"}: rate of positive prediction
#' }
#' @param cdfy a character of the cumulative distribution for the y-axis.
#' Supported values are
#' \itemize{
#' \item \code{"tpr"}: true positive rate
#' \item \code{"tnr"}: true negative rate
#' }
#' @param dspt a decision point (threshold) in [0, 1] for binary classification.
#' If \code{pfmc="acc"}, instances with probabilities <= \code{dspt} are
#' predicted as negative, and those with probabilities > \code{dspt} are
#' predicted as positive. If \code{pfmc="roc"}, \code{dspt} is a threhold on the
#' x-axis of the ROC curve such that the corresponding value on the y-axis is
#' used as the performance metric. For example, if the desired performance
#' metric is the true positive rate at the 5\% false positive rate, specify
#' \code{pfmc="roc"}, \code{cdfx="fpr"}, \code{cdfy="tpr"}, and \code{dspt=0.05}.
#' @param lower a numeric vector containing the minimum values of
#' hyperparameters in the following order:
#' \itemize{
#' \item maximum tree depth
#' \item leaf node size
#' \item bag fraction
#' \item fraction of predictors to try for each split
#' \item shrinkage
#' \item number of trees
#' \item scale of weights for positive cases (for binary classification only)
#' }
#' @param upper a numeric vector containing the maximum values of
#' hyperparameters in the order above.
#' @param quiet a logical of TRUE turns off the display of optimization progress
#' in the console.
#' @return A list of information with the following components:
#' \itemize{
#' \item \code{model}: an ensemble (list) of GBT model(s).
#' \item \code{model_weight}: a vector of model weights whose sum equals
#' \code{ensz}.
#' \item \code{best_idx}: an integer of the iteration index for the best
#' validation performance.
#' \item \code{pred_val}: a matrix of cross-validation predictions where
#' \code{nrow(pred_val) = nrow(x)} and \code{ncol(pred_val) = nitr}.
#' \item \code{perf_val}: a vector of cross-validation performance measures.
#' \item \code{param}: a data.frame of hyperparameter settings visited. Each
#' row of the data.frame is a single hyperparameter setting.
#' \item \code{objective}: a character of the objective function used.
#' \item \code{time} a list of times:
#' \itemize{
#' \item pproc_time a numeric value of preprocessing time in minutes.
#' \item binit_time a numeric value of initialization time in minutes for
#' Bayesian optimization.
#' \item bsrch_time a numeric value of search time in minutes for
#' Bayesian optimization.
#' \item rsrch_time a numeric value of random search time in minutes.
#' \item enslt_time a numeric value of ensemble selection in minutes.
#' \item refit_time a numeric value of refitting (on the full training data)
#' time in minutes.
#' \item total_time a numeric value of the total time in minutes.
#' }
#' \item \code{...}: input arguments (excluding \code{x}, \code{y}, and
#' \code{w}).
#' }
#'
#' @seealso \code{\link{predict.gbts}},
#'          \code{\link{comperf}}
#'
#' @references Rich Caruana, Alexandru Niculescu-Mizil, Geoff Crew, and Alex
#' Ksikes. 2004. Ensemble selection from libraries of models. In Proceedings of
#' the 21st international conference on Machine learning (ICML'04).
#' \url{http://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf}
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
#' yhat_test <- predict(model, test[, pred_idx])
#'
#' # Compute AUC on test data
#' comperf(test[, target_idx], yhat_test, pfmc = "auc")
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
#' yhat_test <- predict(model, test[, pred_idx])
#'
#' # Compute MSE on test data
#' comperf(test[, target_idx], yhat_test, pfmc = "mse")
#'}
#' @export
#' @import stats
#' @import earth
#' @importFrom doRNG %dorng%
#' @importFrom foreach %dopar%
gbts <- function(x, y,
                 w = rep(1, nrow(x)),
                 nitr = 200,
                 nlhs = floor(nitr / 2),
                 nprd = 5000,
                 kfld = 10,
                 srch = c("bayes", "random"),
                 nbst = 100,
                 ensz = 100,
                 nwrk = 2,
                 rpkg = c("gbm"),
                 pfmc = c("acc", "dev", "ks", "auc", "roc", "mse", "rsq", "mae"),
                 cdfx = "fpr",
                 cdfy = "tpr",
                 dspt = 0.5,
                 lower = c(2, 10, 0.1, 0.1, 0.01, 50, 1),
                 upper = c(10, 200, 1, 1, 0.1, 1000, 10),
                 quiet = FALSE) {

  # Check arguments ------------------------------------------------------------

  if (missing(x) || !is.data.frame(x)) {
    stop("Argument 'x' is missing or not a data.frame.")
  }
  if (missing(y) || !is.vector(y)) {
    stop("Argument 'y' is missing or not a vector.")
  }
  if (length(y) != nrow(x)) {
    stop("length(y) != nrow(x).")
  }
  if (length(y) < 2) {
    stop("length(y) < 2.")
  }
  if (length(which(w <= 0)) > 0) {
    stop("Argument 'w' has value(s) <= 0.")
  }
  if (nitr < 1 || !isint(nitr)) {
    stop("Argument 'nitr' < 1 or is not an integer.")
  }
  if (nlhs < 1 || !isint(nlhs)) {
    stop("Argument 'nlhs' < 1 or is not an integer.")
  }
  if (nprd < 1 || !isint(nprd)) {
    stop("Argument 'nprd' < 1 or is not an integer.")
  }
  if (kfld < 2 || !isint(kfld)) {
    stop("Argument 'kfld' < 2 or is not an integer.")
  }
  if (nbst < 2 || !isint(nbst)) {
    stop("Argument 'nbst' < 2 or is not an integer.")
  }
  if (nwrk < 1 || !isint(nwrk)) {
    stop("Argument 'nwrk' < 1 or is not an integer.")
  }
  if (ensz < 1 || !isint(ensz)) {
    stop("Argument 'ensz' < 1 or is not an integer.")
  }
  if (!(srch[1] %in% c("bayes", "random"))) {
    stop("Invalid argument 'srch'.")
  }
  if (!(rpkg[1] %in% c("gbm"))) {
    stop("Invalid argument 'rpkg'.")
  }
  y <- as.numeric(y)
  if (pfmc[1] %in% c("acc", "dev", "ks", "auc", "roc")) {
    if (!identical(sort(unique(y)), c(0, 1))) {
      stop("Response variable 'y' has values other than 0 and 1.")
    }
    if (pfmc[1] == "auc" || pfmc[1] == "roc") {
      if (!(cdfy %in% c("tpr", "tnr"))) {
        stop("Invalid argument 'cdfy'.")
      }
      if (!(cdfx %in% c("fpr", "fnr", "rpp"))) {
        stop("Invalid argument 'cdfx'.")
      }
    }
    if (pfmc[1] == "roc" && !missing(dspt) && !is.null(dspt)) {
      if (length(dspt) > 1) {
        stop("length(dspt) > 1.")
      }
      if (dspt < 0 || dspt > 1) {
        stop("Invalid argument 'dspt'.")
      }
    }
    objective <- "binary:logistic"
  } else if (pfmc[1] %in% c("mse", "rsq", "mae")) {
    if (length(unique(y)) == 1) {
      stop("Response variable 'y' has 1 unique value.")
    }
    objective <- "reg:linear"
  } else {
    stop("Invalid argument 'pfmc'.")
  }
  if (!(is.vector(lower) && length(lower) >= 6)) {
    stop("Invalid argument 'lower'.")
  }
  if (!(is.vector(upper) && length(upper) >= 6)) {
    stop("Invalid argument 'upper'.")
  }
  if (!(isint(lower[1]) && lower[1] >= 1 && isint(upper[1]) &&
        upper[1] > lower[1])) {
    stop("Invalid argument 'lower[1]' and/or 'upper[1]'.")
  }
  if (!(isint(lower[2]) && lower[2] >= 1 && isint(upper[2]) &&
        upper[2] > lower[2] && upper[2] <= length(y))) {
    stop("Invalid argument 'lower[2]' and/or 'upper[2]'.")
  }
  if (!(is.numeric(lower[3]) && lower[3] > 0 && is.numeric(upper[3]) &&
        upper[3] > lower[3] && upper[3] <= 1)) {
    stop("Invalid argument 'lower[3]' and/or 'upper[3]'.")
  }
  if (!(is.numeric(lower[4]) && lower[4] > 0 && is.numeric(upper[4]) &&
        upper[4] > lower[4] && upper[4] <= 1)) {
    stop("Invalid argument 'lower[4]' and/or 'upper[4]'.")
  }
  if (!(is.numeric(lower[5]) && lower[5] > 0 && is.numeric(upper[5]) &&
        upper[5] > lower[5] && upper[5] <= 1)) {
    stop("Invalid argument 'lower[5]' and/or 'upper[5]'.")
  }
  if (!(isint(lower[6]) && lower[6] >= 1 && isint(upper[6]) &&
        upper[6] > lower[6])) {
    stop("Invalid argument 'lower[6]' and/or 'upper[6]'.")
  }
  if (objective == "binary:logistic") {
    if (!(is.numeric(lower[7]) && lower[7] > 0 &&
          is.numeric(upper[7]) && upper[7] > lower[7])) {
      stop("Invalid argument 'lower[7]' and/or 'upper[7]'.")
    }
  }
  max_depth_range <- c(lower[1], upper[1])
  leaf_size_range <- c(lower[2], upper[2])
  bagn_frac_range <- c(lower[3], upper[3])
  coln_frac_range <- c(lower[4], upper[4])
  shrinkage_range <- c(lower[5], upper[5])
  num_trees_range <- c(lower[6], upper[6])
  if (objective == "reg:linear") { scl_pos_w_range <- c(1, 1) }
  else { scl_pos_w_range <- c(lower[7], upper[7]) }

  # Preprocessing --------------------------------------------------------------

  start_time <- Sys.time()
  if (!quiet) { cat("Preprocessing...\n") }

  # Fix coln_frac_range to 1 for gbm
  if (rpkg[1] == "gbm") { coln_frac_range <- 1 }

  # Generate hyperparameter values
  param_range <- rbind(max_depth_range, leaf_size_range, bagn_frac_range,
                       coln_frac_range, shrinkage_range, num_trees_range,
                       scl_pos_w_range)
  sp_idx <- as.vector(apply(param_range, 1, function(z) { z[1] != z[2] }))
  if (length(which(sp_idx)) == 0) { stop("All hyperparameters are fixed.") }
  if (srch[1] == "bayes") {
    param <- propose_lhs_param(nlhs, max_depth_range,
                               leaf_size_range, bagn_frac_range,
                               coln_frac_range, shrinkage_range,
                               num_trees_range, scl_pos_w_range)
  } else {
    param <- propose_random_param(nitr, max_depth_range,
                                  leaf_size_range, bagn_frac_range,
                                  coln_frac_range, shrinkage_range,
                                  num_trees_range, scl_pos_w_range)
  }

  # Adjust leaf size to be <= the size of the training set / 2
  trn_size <- floor((nrow(x) / kfld)) * (kfld - 1)
  param$leaf_size <- apply(param[c("leaf_size", "bagn_frac")], 1,
                           function(u) { min(u[1], floor((u[2]*trn_size-2)/2)) })

  # Determine cross-validation (CV) partition indicies
  start_idx <- floor((nrow(x) / kfld) * 1:kfld - (nrow(x) / kfld) + 1)
  end_idx <- start_idx[2:length(start_idx)] - 1
  end_idx <- c(end_idx, nrow(x))

  # Record preprocessing time
  pproc_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  if (!quiet) {
    cat(paste0("Completed preprocessing in ", round(pproc_time, 6), " seconds\n"))
  }

  # Start hyperparameter search ------------------------------------------------

  j <- 1
  initn_start_time <- Sys.time()
  if (!quiet) {
    if (srch[1] == "bayes") {
      cat("Initializing the performance model for Bayesian optimization...\n")
    } else {
      cat("Performing random search...\n")
    }
  }

  # Parellel backend setup
  allcores <- parallel::detectCores() - foreach::getDoParWorkers()
  if (nwrk > allcores) {
    warning(paste("Requested", nwrk, "workers, but using the available",
                  allcores, "only."))
    nwrk <- allcores
  }
  cl <- parallel::makeCluster(nwrk)
  doParallel::registerDoParallel(cl)

  # Train models and predict on validation sets
  work_idx <- expand.grid(1:kfld, 1:nrow(param))
  pred_val_cv <- foreach::foreach(j = 1:nrow(work_idx), .options.RNG = 123) %dorng% {
    val_idx <- start_idx[work_idx[j, 1]]:end_idx[work_idx[j, 1]]
    model <- gbt_fit(x = x[-val_idx, ], y = y[-val_idx], w = w[-val_idx],
                     max_depth = param$max_depth[work_idx[j, 2]],
                     leaf_size = param$leaf_size[work_idx[j, 2]],
                     bagn_frac = param$bagn_frac[work_idx[j, 2]],
                     coln_frac = param$coln_frac[work_idx[j, 2]],
                     shrinkage = param$shrinkage[work_idx[j, 2]],
                     num_trees = param$num_trees[work_idx[j, 2]],
                     scl_pos_w = param$scl_pos_w[work_idx[j, 2]],
                     rpkg = rpkg[1], objective = objective)
    gbt_predict(model, x[val_idx, ], rpkg[1], type = "response")
  }
  pred_val <- matrix(unlist(pred_val_cv), nrow = nrow(x), ncol = nrow(param))

  # Compute validation performance
  perf_val <- apply(pred_val, 2, function(u) {
    comperf(y = y, yhat = u, w = w, pfmc = pfmc[1], cdfy = cdfy, cdfx = cdfx,
            dspt = dspt) })
  fbp <- find_best_perf(perf_val, pfmc[1])
  best_perf_val <- fbp$best_perf
  best_idx <- fbp$best_idx

  # Record random search time
  rsrch_time <- NA
  if (srch[1] == "random") {
    binit_time <- NA
    bsrch_time <- NA
    rsrch_time <- as.numeric(difftime(Sys.time(), initn_start_time, units = "mins"))
    if (!quiet) {
      cat(paste0("Completed random search in ", round(rsrch_time, 2), " minutes\n"))
    }
  }

  if (srch[1] == "bayes" && nitr > nlhs) {

    # Initialize model of performance
    perf_data <- data.frame(Y = perf_val, param[, sp_idx])
    perf_fits <- foreach::foreach(b = 1:nbst, .options.RNG = 456) %dorng% {
      bst_idx <- sample(1:nrow(perf_data), nrow(perf_data), replace = TRUE)
      earth::earth(Y ~ ., data = perf_data[bst_idx, ],
                   degree = min(ncol(perf_data), 3), nk = 100)
    }

    # Record initialization time
    binit_time <- as.numeric(difftime(Sys.time(), initn_start_time, units = "mins"))
    if (!quiet) {
      cat(paste0("Completed initialization in ", round(binit_time, 2), " minutes\n"))
    }

    # Extend variables
    pred_val <- cbind(pred_val, matrix(NA, nrow = nrow(x), ncol = (nitr-nlhs)))
    perf_val <- c(perf_val, rep(NA, (nitr-nlhs)))

    bayes_start_time <- Sys.time()
    if (!quiet) { cat("Start Bayesian optimization\n") }

    for (trial_idx in (nlhs + 1):nitr) {
      trial_start_time <- Sys.time()

      # Propose candidate hyperparameter values
      Xs <- propose_lhs_param(nprd, max_depth_range = max_depth_range,
                              leaf_size_range = leaf_size_range,
                              bagn_frac_range = bagn_frac_range,
                              coln_frac_range = coln_frac_range,
                              shrinkage_range = shrinkage_range,
                              num_trees_range = num_trees_range,
                              scl_pos_w_range = scl_pos_w_range)

      # Adjust leaf size to be <= size of the training set / 2
      Xs$leaf_size <- apply(Xs[c("leaf_size", "bagn_frac")], 1,
                            function(u) { min(u[1], floor((u[2]*trn_size-2)/2)) })

      # Predict performance for the candidate hyperparameters
      b <- 1
      perf_cnd_list <- foreach::foreach(b = 1:nbst) %dorng% {
        predict(perf_fits[[b]], Xs[, sp_idx])
      }
      perf_cnd <- matrix(unlist(perf_cnd_list), nrow = nprd, ncol = nbst)

      # Compute expected improvement
      if (pfmc[1] %in% c("dev", "mse", "mae")) {
        c_min <- best_perf_val
      } else if (pfmc[1] %in% c("acc", "ks", "auc", "roc", "rsq")) {
        perf_cnd[perf_cnd < 0] <- 0
        perf_cnd[perf_cnd > 1] <- 1
        perf_cnd <- log((1 - perf_cnd) / perf_cnd)
        c_min <- log((1 - best_perf_val) / best_perf_val)
      } else {
        stop("Invalid 'pfmc'.")
      }
      c_mean <- apply(perf_cnd, 1, mean)
      c_sd <- apply(perf_cnd, 1, sd)
      u <- (c_min - c_mean) / c_sd
      ei <- c_sd * (u * pnorm(u) + dnorm(u))
      max_ei_idx <- which.max(ei)

      # Train models and predict on validation sets
      pred_val_cv_i <- foreach::foreach(j = 1:kfld, .options.RNG = 123) %dorng% {
        val_idx <- start_idx[j]:end_idx[j]
        model <- gbt_fit(x = x[-val_idx, ], y = y[-val_idx], w = w[-val_idx],
                         max_depth = Xs$max_depth[max_ei_idx],
                         leaf_size = Xs$leaf_size[max_ei_idx],
                         bagn_frac = Xs$bagn_frac[max_ei_idx],
                         coln_frac = Xs$coln_frac[max_ei_idx],
                         shrinkage = Xs$shrinkage[max_ei_idx],
                         num_trees = Xs$num_trees[max_ei_idx],
                         scl_pos_w = Xs$scl_pos_w[max_ei_idx],
                         rpkg = rpkg[1], objective = objective)
        gbt_predict(model, x[val_idx, ], rpkg[1], type = "response")
      }
      pred_val_i <- as.numeric(unlist(pred_val_cv_i))

      # Compute validation performance
      perf_val_i <- comperf(y = y, yhat = pred_val_i, w = w, pfmc = pfmc[1],
                            cdfy = cdfy, cdfx = cdfx, dspt = dspt)
      pred_val[, trial_idx] <- pred_val_i
      perf_val[trial_idx] <- perf_val_i
      fbp <- find_best_perf(perf_val[1:trial_idx], pfmc[1])
      best_perf_val <- fbp$best_perf
      best_idx <- fbp$best_idx

      # Save the selected hyperparameters
      param[trial_idx, ] <- c(Xs$max_depth[max_ei_idx], Xs$leaf_size[max_ei_idx],
                              Xs$bagn_frac[max_ei_idx], Xs$coln_frac[max_ei_idx],
                              Xs$shrinkage[max_ei_idx], Xs$num_trees[max_ei_idx],
                              Xs$scl_pos_w[max_ei_idx])

      # Update the model of performance
      perf_data <- data.frame(Y = perf_val[1:trial_idx], param[, sp_idx])
      perf_fits <- foreach::foreach(b = 1:nbst, .options.RNG = 456) %dorng% {
        bst_idx <- sample(1:nrow(perf_data), nrow(perf_data), replace = TRUE)
        earth::earth(Y ~ ., data = perf_data[bst_idx, ],
                     degree = min(ncol(perf_data), 3), nk = 100)
      }

      # Update progress
      trial_time <- as.numeric(difftime(Sys.time(), trial_start_time, units = "mins"))
      bsrch_time <- as.numeric(difftime(Sys.time(), bayes_start_time, units = "mins"))
      if (!quiet) {
        cat(paste0("[", trial_idx, "] ", pfmc[1], ": ",
                   round(perf_val[trial_idx], 4), " validation, ",
                   round(best_perf_val, 4), " best | time (mins): ",
                   round(trial_time, 2), " trial, ", round(bsrch_time, 2),
                   " total\n"))
      }
    }

    # Record Bayesian optimization time
    if (!quiet) {
      cat(paste0("Completed Bayesian optimization in ", round(bsrch_time, 2),
                 " minutes\n"))
    }
  }

  # Build final model(s) -------------------------------------------------------

  # Ensemble Selection
  enslt_start_time <- Sys.time()
  if (!quiet) { cat("Performing ensemble selection...\n") }
  pred_ensbl <- 0.5
  perf_ensbl <- rep(NA, ensz)
  ensbl_idx <- rep(NA, ensz)
  if (objective == "binary:logistic") { logit_pred_val <- logit(pred_val) }
  for (i in 1:ensz) {
    if (objective == "binary:logistic") {
      # pred_val and pred_ensbl are on the probability scale
      pred_ensbl_i <- inv_logit((logit_pred_val + logit(pred_ensbl) * (i - 1)) / i)
    } else {
      pred_ensbl_i <- (pred_val + pred_ensbl * (i - 1)) / i
    }
    pfms_en_i <- apply(pred_ensbl_i, 2, function(u) {
      comperf(y = y, yhat = u, w = w, pfmc = pfmc[1], cdfy = cdfy, cdfx = cdfx,
              dspt = dspt) })
    fbp <- find_best_perf(pfms_en_i, pfmc)
    perf_ensbl[i] <- fbp$best_perf
    ensbl_idx[i] <- fbp$best_idx
    pred_ensbl <- pred_ensbl_i[, fbp$best_idx]
  }

  # Record ensemble selection time
  enslt_time <- as.numeric(difftime(Sys.time(), enslt_start_time, units = "mins"))
  if (!quiet) {
    cat(paste0("Completed ensemble selection in ", round(enslt_time, 2),
               " minutes\n"))
  }

  # Fit model(s) on the full training set
  refit_start_time <- Sys.time()
  if (!quiet) { cat("Refit selected models on the full training set...\n") }
  unique_ensbl_idx <- sort(unique(ensbl_idx))
  ensbl_param <- param[unique_ensbl_idx, , drop = FALSE]
  model <- foreach::foreach(j = 1:nrow(ensbl_param), .options.RNG = 123) %dorng% {
    if (nrow(ensbl_param) == 1) { set.seed(123) }
    gbt_fit(x = x, y = y, w = w,
            max_depth = ensbl_param$max_depth[j],
            leaf_size = ensbl_param$leaf_size[j],
            bagn_frac = ensbl_param$bagn_frac[j],
            coln_frac = ensbl_param$coln_frac[j],
            shrinkage = ensbl_param$shrinkage[j],
            num_trees = ensbl_param$num_trees[j],
            scl_pos_w = ensbl_param$scl_pos_w[j],
            rpkg = rpkg[1],
            objective = objective)
  }
  model_weight <- as.numeric(ftable(ensbl_idx))
  parallel::stopCluster(cl)

  # Record refitting time
  refit_time <- as.numeric(difftime(Sys.time(), refit_start_time, units = "mins"))
  total_time <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  if (!quiet) {
    cat(paste0("Completed refitting in ", round(refit_time, 2), " minutes\n"))
    cat(paste0("Total time: ", round(total_time, 2), " minutes\n"))
  }

  time <- list(pproc_time = pproc_time,
               binit_time = binit_time,
               bsrch_time = bsrch_time,
               rsrch_time = rsrch_time,
               enslt_time = enslt_time,
               refit_time = refit_time,
               total_time = total_time)
  output <- list(model = model,
                 model_weight = model_weight,
                 best_idx = best_idx,
                 pred_val = pred_val,
                 perf_val = perf_val,
                 param = param,
                 objective = objective[1],
                 time = time,
                 nitr = nitr,
                 nlhs = nlhs,
                 nprd = nprd,
                 kfld = kfld,
                 nbst = nbst,
                 nwrk = nwrk,
                 ensz = ensz,
                 srch = srch[1],
                 rpkg = rpkg[1],
                 pfmc = pfmc[1],
                 cdfx = cdfx,
                 cdfy = cdfy,
                 dspt = dspt,
                 lower = lower,
                 upper = upper)
  class(output) <- "gbts"
  return(output)
}


#' Predict method for ensemble of Gradient Boosted Trees
#'
#' This function generates predictions by weighted averaging the predictions
#' from each model in the ensemble returned from \code{\link{gbts}}. Weighted
#' average is computed on the log-odds scale for binary classification.
#'
#' @param object a model object returned from \code{\link{gbts}}.
#' @param x a data.frame of predictors. It must follow the same format as the
#' training dataset on which the model \code{object} is developed.
#' @param nwrk an integer of the number of computing workers to be used. If
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
#' @method predict gbts
predict.gbts <- function(object, x, nwrk = 2, ...) {
  # TODO: update documentation for ensemble
  if (class(object) != "gbts") { stop("'object' is not of class 'gbts'") }

  # Parellel backend setup
  allcores <- parallel::detectCores() - foreach::getDoParWorkers()
  if (nwrk > allcores) {
    warning(paste("Requested", nwrk, "workers, but using the available",
                  allcores, "only."))
    nwrk <- allcores
  }
  cl <- parallel::makeCluster(nwrk)
  doParallel::registerDoParallel(cl)

  # Make predictions
  j <- 1
  nmod <- length(object$model)
  pred <- foreach::foreach(j = 1:nmod, .options.RNG = 123) %dorng% {
    gbt_predict(object$model[[j]], x, object$rpkg, type = "response")
  }
  parallel::stopCluster(cl)
  pred <- matrix(unlist(pred), nrow = nrow(x), ncol = nmod)

  # Apply model weights to predictions
  ws <- sum(object$model_weight)
  if (object$objective == "binary:logistic") {
    return(apply(pred, 1, function(u) {
      inv_logit(sum(logit(u) * object$model_weight) / ws) }))
  } else {
    return(apply(pred, 1, function(u) { sum(u * object$model_weight) / ws }))
  }
}


# Propose hyperparameters uniformly at random
propose_random_param <-
  function(n, max_depth_range = NULL,
           leaf_size_range = NULL, bagn_frac_range = NULL,
           coln_frac_range = NULL, shrinkage_range = NULL,
           num_trees_range = NULL, scl_pos_w_range = NULL) {
    param <- list()

    # max_depth
    if (length(max_depth_range) == 1) {
      param[["max_depth"]] <- rep(max_depth_range, n)
    } else {
      param[["max_depth"]] <- floor(runif(n, max_depth_range[1], max_depth_range[2]))
    }

    # leaf_size
    if (length(leaf_size_range) == 1) {
      param[["leaf_size"]] <- rep(leaf_size_range, n)
    } else {
      param[["leaf_size"]] <- floor(runif(n, leaf_size_range[1], leaf_size_range[2]))
    }

    # bagn_frac
    if (length(bagn_frac_range) == 1) {
      param[["bagn_frac"]] <- rep(bagn_frac_range, n)
    } else {
      param[["bagn_frac"]] <- runif(n, bagn_frac_range[1], bagn_frac_range[2])
    }

    # coln_frac
    if (length(coln_frac_range) == 1) {
      param[["coln_frac"]] <- rep(coln_frac_range, n)
    } else {
      param[["coln_frac"]] <- runif(n, coln_frac_range[1], coln_frac_range[2])
    }

    # shrinkage
    if (length(shrinkage_range) == 1) {
      param[["shrinkage"]] <- rep(shrinkage_range, n)
    } else {
      param[["shrinkage"]] <- runif(n, shrinkage_range[1], shrinkage_range[2])
    }

    # num_trees
    if (length(num_trees_range) == 1) {
      param[["num_trees"]] <- rep(num_trees_range, n)
    } else {
      param[["num_trees"]] <- floor(runif(n, num_trees_range[1], num_trees_range[2]))
    }

    # scl_pos_w
    if (length(scl_pos_w_range) == 1) {
      param[["scl_pos_w"]] <- rep(scl_pos_w_range, n)
    } else {
      param[["scl_pos_w"]] <- runif(n, scl_pos_w_range[1], scl_pos_w_range[2])
    }

    return(data.frame(param))
  }


# Propose hyperparameters using Latin Hypercube Sampling
propose_lhs_param <-
  function(nlhs, max_depth_range = NULL,
           leaf_size_range = NULL, bagn_frac_range = NULL,
           coln_frac_range = NULL, shrinkage_range = NULL,
           num_trees_range = NULL, scl_pos_w_range = NULL) {
    # Find out which hyperparameter(s) to search
    param_range <- rbind(max_depth_range, leaf_size_range, bagn_frac_range,
                         coln_frac_range, shrinkage_range, num_trees_range,
                         scl_pos_w_range)
    sp_idx <- as.vector(apply(param_range, 1, function(z) { z[1] != z[2] }))
    if (length(which(sp_idx)) == 0) { stop("All hyperparameters are fixed.") }
    nfix <- length(which(!sp_idx))

    # Generate hyperparameters using Latin Hypercube sampling
    param <- matrix(NA, nrow = nlhs, ncol = 7)
    rang <- as.numeric(apply(param_range[sp_idx, ], 1, diff))
    minv <- as.numeric(apply(param_range[sp_idx, ], 1, min))
    param[, sp_idx] <- t(t(randomLHS(n = nlhs, k = length(which(sp_idx)))) *
                           rang + minv)
    param[, !sp_idx] <- matrix(rep(param_range[!sp_idx, 1], each = nlhs),
                               nrow = nlhs, ncol = nfix)
    param <- as.data.frame(param)
    names(param) <- c("max_depth", "leaf_size", "bagn_frac", "coln_frac",
                      "shrinkage", "num_trees", "scl_pos_w")
    param$max_depth <- floor(param$max_depth)
    param$leaf_size <- floor(param$leaf_size)
    param$num_trees <- floor(param$num_trees)
    return(param)
  }


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

    # gbm
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
  } else {
    stop(paste("Invalid 'rpkg':", rpkg[1]))
  }
  return(pred)
}
