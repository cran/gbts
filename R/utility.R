#' Compute model performance
#'
#' This function computes model performance given a vector of response values
#' and a vector of predictions.
#'
#' @param y a vector of numeric response values.
#' @param yhat a vector of model predictions.
#' @param w an optional vector of observation weights.
#' @param pfmc a character of the performance metric. For binary classification,
#' \code{pfmc} accepts:
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
#' @return A single or a vector of numeric values of model performance, or a
#' list of two components \code{x} and \code{y} representing the ROC curve.
#'
#' @seealso \code{\link{gbts}},
#'          \code{\link{predict.gbts}}
#'
#' @author Waley W. J. Liang <\email{wliang10@gmail.com}>
#'
#' @examples
#' y = c(0, 1, 0, 1, 1, 1)
#' yhat = c(0.5, 0.9, 0.2, 0.7, 0.6,  0.4)
#' comperf(y, yhat, pfmc = "auc")
#' # 0.875
#'
#' y = 1:10
#' yhat = c(1:5 - 0.1, 6:10 + 0.1)
#' comperf(y, yhat, pfmc = "mse")
#' # 0.01
#'
#' @export
comperf <- function(y, yhat, w = rep(1, length(y)), pfmc = NULL, cdfx = "fpr",
                    cdfy = "tpr", dspt = 0.5) {
  if (missing(y) || !is.vector(y)) { stop("'y' is missing or not a vector.") }
  if (missing(yhat) || !is.vector(yhat)) { stop("'yhat' is missing or not a vector.") }
  if (length(y) != length(yhat)) { stop("'y' and 'yhat' have different lengths.") }
  if (length(which(w <= 0)) > 0) { stop("'w' has value(s) <= 0.") }

  y <- as.numeric(y)
  if (pfmc %in% c("acc", "dev", "ks", "auc", "roc")) { # Binary classification

    # Check target variable
    if (!identical(sort(unique(y)), c(0, 1))) {
      stop("Response variable 'y' has value(s) other than 0 and 1.")
    }

    # Check cdfx and cdfy
    if (pfmc == "auc" || pfmc == "roc") {
      if (!(cdfx %in% c("fpr", "fnr", "rpp"))) { stop("Invalid 'cdfx'.") }
      if (!(cdfy %in% c("tpr", "tnr"))) { stop("Invalid 'cdfy'.") }
    }

    # Check operating points
    if (pfmc == "roc" && !missing(dspt) && !is.null(dspt)) {
      oob <- length(which(dspt < 0 || dspt > 1))
      if (oob > 0) { stop("'dspt' has invalid value(s).") }
    }

    # Compute metadata
    counts <- ftable(xtabs(cbind(w * (1 - y), w * y) ~ yhat))
    uprd <- as.numeric(rownames(as.table(counts))) # unique predictions
    wcn <- counts[, 1] # Weighted counts for negatives
    wcp <- counts[, 2] # Weighted counts for positives
    wca <- wcn + wcp # Weighted counts for all cases
    nn <- cumsum(wcn) # Cumulative counts for negatives
    np <- cumsum(wcp) # Cumulative counts for positives
    na <- nn + np # Cumulative counts for all cases
    m <- nrow(counts)
    pn <- c(0, nn / nn[m]) # Cumulative distributions for negatives
    pp <- c(0, np / np[m]) # Cumulative distributions for positives
    pa <- c(0, na / na[m]) # Cumulative distributions for all cases
    mtdt <- list(y = y, yhat = yhat, w = w, wcn = wcn, wcp = wcp, wca = wca,
                 nn = nn, np = np, na = na, pn = pn, pp = pp, pa = pa,
                 uprd = uprd)
  } else if (pfmc %in% c("mse", "rsq", "mae")) { # Regression
    if (length(unique(y)) <= 1) {
      stop("Response variable 'y' has only 1 unique value.")
    }

    # Compute metadata
    res <- y - yhat # Residuals
    rss <- sum(w * (res)^2) # Residual sum of squares
    rsa <- sum(w * abs(res)) # Residual sum of absolute values
    sow <- sum(w) # Sum of weights
    mtdt <- list(y = y, yhat = yhat, w = w, rss = rss, rsa = rsa, sow = sow)
  } else {
    stop("Argument 'pfmc' is either missing or not supported.")
  }

  # Compute performance
  func <- utils::getFromNamespace(pfmc, ns = "gbts")
  if (pfmc %in% c("dev", "ks", "mse", "rsq", "mae")) {
    return(func(mtdt))
  } else if (pfmc == "acc") {
    return(func(mtdt, dspt))
  } else {
    return(func(mtdt, cdfx, cdfy, dspt))
  }
}


# Compute accuracy
#
# This function computes accuracy for binary classification.
#
# @param mtdt a list of metadata used for performance computation.
# @param dspt a decision point (threshold) in [0, 1]. Instances with
# probabilities <= \code{dspt} are predicted as negative, and those with
# probabilities > \code{dspt} are predicted as positive.
# @return A single or a vector of numeric values (sorted in ascending order of
# predictions) of accuracy.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
acc <- function(mtdt, dspt = 0.5) {
  ac <- (max(mtdt$np) + mtdt$nn - mtdt$np) / max(mtdt$na)
  ac <- c(max(mtdt$np) / max(mtdt$na), ac)
  if (min(mtdt$uprd) >= 0 && max(mtdt$uprd) <= 1) {
    if (max(mtdt$uprd) <= dspt) { dspt <- max(mtdt$uprd) } # Predictions = 0
    else if (min(mtdt$uprd) > dspt) { dspt <- 0 } # Predictions = 1
  } else {
    stop("Prediction(s) outside of [0, 1].")
  }

  # Compute proportion of samples below 'dspt'
  rt <- approx(x = c(0, mtdt$uprd), y = mtdt$pa, xout = dspt, ties = "ordered")$y

  # Compute the accuracy
  ac <- approx(x = mtdt$pa, y = ac, xout = rt, ties = "ordered")$y
  if (sum(is.na(ac)) > 0) { stop("Approximation is out of range.") }
  return(ac)
}


# Compute deviance
#
# This function computes deviance for binary classification.
#
# @param mtdt a list of metadata used for performance computation.
# @param epsilon precision used to stabilize computation.
# @return A single numeric value of deviance.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
dev <- function(mtdt, epsilon = .Machine$double.eps) {
  p <- mtdt$uprd
  p[p < epsilon] <- epsilon
  p[p > (1 - epsilon)] <- (1 - epsilon)
  ll <- sum(mtdt$wcp * log(p) + mtdt$wcn * log(1 - p))
  return(-2 * ll / sum(mtdt$w))
}


# Compute Kolmogorov-Smirnov statistic
#
# This function computes the Kolmogorov-Smirnov statistic for binary
# classification.
#
# @param mtdt a list of metadata used for performance computation.
# @return A single numeric value of the Kolmogorov-Smirnov statistic.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
ks <- function(mtdt) {
  return(max(abs(mtdt$pn - mtdt$pp)))
}


# Compute the receiver operating characteristic curve
#
# This function computes the receiver operating characteristic curve.
#
# @param mtdt a list of metadata used for performance computation.
# @param cdfx a character of the cumulative distribution for the x-axis.
# Supported values are
# \itemize{
# \item \code{"fpr"}: false positive rate
# \item \code{"fnr"}: false negative rate
# \item \code{"rpp"}: rate of positive prediction
# }
# @param cdfy a character of the cumulative distribution for the y-axis.
# Supported values are
# \itemize{
# \item \code{"tpr"}: true positive rate
# \item \code{"tnr"}: true negative rate
# }
# @param dspt decision point (threshold) in [0, 1] on the x-axis at which to
# evaluate the ROC curve.
# @return A ROC curve contained in a list of two components:
# \itemize{
# \item \code{x}: a vector of \code{cdfx} values.
# \item \code{y}: a vector of \code{cdfy} values.
# }
# OR a single/vector of evaluation(s) of the ROC curve at \code{dspt}.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
roc <- function(mtdt, cdfx = "fpr", cdfy = "tpr", dspt) {
  # Cumulative distribution for the x-axis
  if (cdfx == "fpr") { x <- rev(1 - mtdt$pn) } # In descending order of predictions
  else if (cdfx == "fnr") { x <- mtdt$pp } # In ascending order of predictions
  else if (cdfx == "rpp") { x <- rev(1 - mtdt$pa) } # In descending order of predictions
  else { stop("Invalid 'cdfx'.") }

  # Cumulative distribution for the y-axis
  if (cdfy == "tpr") { y <- 1 - mtdt$pp }
  else if (cdfy == "tnr") { y <- mtdt$pn }
  else { stop("Invalid 'cdfy'.") }

  # Sort y in descending order of predictions
  if (cdfx == "fpr" || cdfx == "rpp") { y <- rev(y) }

  if (missing(dspt) || is.null(dspt)) {
    # Return the entire ROC curve
    return(list(x = x, y = y))
  } else {
    # Return evaluation(s) of the ROC curve at the threshold(s)
    ap <- approx(x = x, y = y, xout = dspt, ties = "ordered")$y
    if (sum(is.na(ap)) > 0) { warning("'dspt' out of bound.") }
    return(ap)
  }
}


# Compute area under the ROC curve
#
# This function computes area under the ROC curve for binary classification.
#
# @param mtdt a list of metadata used for performance computation.
# @param cdfx a character of the cumulative distribution for the x-axis. See
# \code{roc()} for the supported values.
# @param cdfy a character of the cumulative distribution for the y-axis. See
# \code{roc()} for the supported values.
# @return A single numeric value of the area under the ROC curve.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
auc <- function(mtdt, cdfx = "fpr", cdfy = "tpr", ...) {
  r <- roc(mtdt, cdfx, cdfy)
  return(trapz(r$x, r$y))
}


# Compute mean squared error
#
# This function computes the mean squared error.
#
# @param mtdt a list of metadata used for performance computation.
# @return A single numeric value of mean squared error.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
mse <- function(mtdt) {
  return(mtdt$rss / mtdt$sow)
}


# Compute mean absolute error
#
# This function computes the mean absolute error.
#
# @param mtdt a list of metadata used for performance computation.
# @return A single numeric value of mean absolute error.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
mae <- function(mtdt) {
  return(mtdt$rsa / mtdt$sow)
}


# Compute r-squared (coefficient of determination)
#
# This function computes the r-squared (coefficient of determination).
#
# @param mtdt a list of metadata used for performance computation.
# @return A single numeric value of r-squared.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
rsq <- function(mtdt) {
  ybar <- sum(mtdt$w * mtdt$y) / mtdt$sow
  sst <- sum(mtdt$w * (mtdt$y - ybar)^2)
  return(1 - mtdt$rss / sst)
}


# Find the best performance measures
#
# This function returns the best performance measure within a vector of
# performance measures.
#
# @param pfms a vector of performance measures.
# @param pfmc a character of the performance metric
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
find_best_perf <- function(pfms, pfmc) {
  if (pfmc[1] %in% c("dev", "mse", "mae")) {
    return(list(best_perf = min(pfms)[1], best_idx = which.min(pfms)[1]))
  } else if (pfmc[1] %in% c("acc", "ks", "auc", "roc", "rsq")) {
    return(list(best_perf = max(pfms)[1], best_idx = which.max(pfms)[1]))
  }
}


# Compute average of probabilities on the logit scale
#
# This function computes the average of probabilities on the logit scale.
#
# @param x a vector of probabilities.
# @return A single numeric value.
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
lgmean <- function(x) {
  return(inv_logit(mean(logit(x))))
}


# Check whether a number is an integer
#
# This function checks whether a number is an integer.
#
# @param x a numeric value.
# @return A logical (TRUE or FALSE).
#
# @author Waley W. J. Liang <\email{wliang10@gmail.com}>
isint <- function(x) {
  return(as.integer(x) == x)
}


# Integration using trapezoidal rule (from the 'caTools' package)
trapz <- function (x, y) {
  idx = 2:length(x)
  return(as.double((x[idx] - x[idx - 1]) %*% (y[idx] + y[idx - 1]))/2)
}


# Logit (from the 'gtools' package)
logit <- function (x, min = 0, max = 1) {
  p <- (x - min) / (max - min)
  log(p / (1 - p))
}


# Inverse logit (from the 'gtools' package)
inv_logit <- function (x, min = 0, max = 1) {
  p <- exp(x) / (1 + exp(x))
  p <- ifelse(is.na(p) & !is.na(x), 1, p)
  p * (max - min) + min
}


# Latin Hypercube Sampling (from the 'lhs' package)
randomLHS <- function (n, k, preserveDraw = FALSE) {
  if (length(n) != 1 | length(k) != 1)
    stop("n and k may not be vectors")
  if (any(is.na(c(n, k))))
    stop("n and k may not be NA or NaN")
  if (any(is.infinite(c(n, k))))
    stop("n and k may not be infinite")
  if (floor(n) != n | n < 1)
    stop("n must be a positive integer\n")
  if (floor(k) != k | k < 1)
    stop("k must be a positive integer\n")
  if (!(preserveDraw %in% c(TRUE, FALSE)))
    stop("preserveDraw must be TRUE/FALSE")
  if (preserveDraw) {
    f <- function(X, N) order(runif(N)) - 1 + runif(N)
    P <- sapply(1:k, f, N = n)
  } else {
    ranperm <- function(X, N) order(runif(N))
    P <- matrix(nrow = n, ncol = k)
    P <- apply(P, 2, ranperm, N = n)
    P <- P - 1 + matrix(runif(n * k), nrow = n, ncol = k)
  }
  return(P/n)
}


# Truncated Normal distribution (from the 'msm' package)
rtnorm <- function (n, mean = 0, sd = 1, lower = -Inf, upper = Inf) {
  if (length(n) > 1)
    n <- length(n)
  mean <- rep(mean, length = n)
  sd <- rep(sd, length = n)
  lower <- rep(lower, length = n)
  upper <- rep(upper, length = n)
  lower <- (lower - mean)/sd
  upper <- (upper - mean)/sd
  ind <- seq(length = n)
  ret <- numeric(n)
  alg <- ifelse(lower > upper, -1, ifelse(((lower < 0 & upper ==
    Inf) | (lower == -Inf & upper > 0) | (is.finite(lower) &
    is.finite(upper) & (lower < 0) & (upper > 0) & (upper -
    lower > sqrt(2 * pi)))), 0, ifelse((lower >= 0 & (upper >
    lower + 2 * sqrt(exp(1))/(lower + sqrt(lower^2 + 4)) *
    exp((lower * 2 - lower * sqrt(lower^2 + 4))/4))),
    1, ifelse(upper <= 0 & (-lower > -upper + 2 * sqrt(exp(1))/(-upper +
    sqrt(upper^2 + 4)) * exp((upper * 2 - -upper * sqrt(upper^2 +
    4))/4)), 2, 3))))
  ind.nan <- ind[alg == -1]
  ind.no <- ind[alg == 0]
  ind.expl <- ind[alg == 1]
  ind.expu <- ind[alg == 2]
  ind.u <- ind[alg == 3]
  ret[ind.nan] <- NaN
  while (length(ind.no) > 0) {
    y <- rnorm(length(ind.no))
    done <- which(y >= lower[ind.no] & y <= upper[ind.no])
    ret[ind.no[done]] <- y[done]
    ind.no <- setdiff(ind.no, ind.no[done])
  }
  stopifnot(length(ind.no) == 0)
  while (length(ind.expl) > 0) {
    a <- (lower[ind.expl] + sqrt(lower[ind.expl]^2 + 4))/2
    z <- rexp(length(ind.expl), a) + lower[ind.expl]
    u <- runif(length(ind.expl))
    done <- which((u <= exp(-(z - a)^2/2)) & (z <= upper[ind.expl]))
    ret[ind.expl[done]] <- z[done]
    ind.expl <- setdiff(ind.expl, ind.expl[done])
  }
  stopifnot(length(ind.expl) == 0)
  while (length(ind.expu) > 0) {
    a <- (-upper[ind.expu] + sqrt(upper[ind.expu]^2 + 4))/2
    z <- rexp(length(ind.expu), a) - upper[ind.expu]
    u <- runif(length(ind.expu))
    done <- which((u <= exp(-(z - a)^2/2)) & (z <= -lower[ind.expu]))
    ret[ind.expu[done]] <- -z[done]
    ind.expu <- setdiff(ind.expu, ind.expu[done])
  }
  stopifnot(length(ind.expu) == 0)
  while (length(ind.u) > 0) {
    z <- runif(length(ind.u), lower[ind.u], upper[ind.u])
    rho <- ifelse(lower[ind.u] > 0, exp((lower[ind.u]^2 -
      z^2)/2), ifelse(upper[ind.u] < 0, exp((upper[ind.u]^2 -
      z^2)/2), exp(-z^2/2)))
    u <- runif(length(ind.u))
    done <- which(u <= rho)
    ret[ind.u[done]] <- z[done]
    ind.u <- setdiff(ind.u, ind.u[done])
  }
  stopifnot(length(ind.u) == 0)
  ret * sd + mean
}
