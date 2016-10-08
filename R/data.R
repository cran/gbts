#' German credit data
#'
#' This dataset classifies 1,000 people described by a set of attributes as
#' good or bad credit risks.
#'
#' @format A list of 4 components:
#' \describe{
#'   \item{train}{A \code{data.frame} of the training dataset which contains 700
#'   rows and 21 columns.}
#'   \item{test}{A \code{data.frame} of the test dataset which contains 300 rows
#'   and 21 columns.}
#'   \item{target_idx}{A column index of the target (response) varible.}
#'   \item{pred_idx}{A set of column indicies of the predictors.}
#' }
#' @source \url{https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)}
"german_credit"

#' Boston housing data
#'
#' This dataset concerns the values of 506 houses in suburbs of Boston.
#'
#' @format A list of 4 components:
#' \describe{
#'   \item{train}{A \code{data.frame} of the training dataset which contains 354
#'   rows and 14 columns.}
#'   \item{test}{A \code{data.frame} of the test dataset which contains 152 rows
#'   and 14 columns.}
#'   \item{target_idx}{A column index of the target (response) varible.}
#'   \item{pred_idx}{A set of column indicies of the predictors.}
#' }
#' @source \url{https://archive.ics.uci.edu/ml/datasets/Housing}
"boston_housing"
