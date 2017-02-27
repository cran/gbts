# gbts 1.2.0

* Implemented ensemble selection to construct an ensemble of models in the 
output of gbts().

* Revised the API of gbts() to simplify the specification of minimum and maximum 
values of hyperparameters.

* Improved the display of optimization progress.

* Terminated support for the R package "xgboost".

# gbts 1.0.1

* Modified access to gbm predict() to be compatible with the current and next 
version of gbm.

* Revised the documentation of gbts() and the DESCRIPTION file to describe 
Bayesian optimization in replacement of active learning. This is merely a
documentation change for the same algorithm.

* Allowed the "srch" argument of gbts() to accept "bayes" for Bayesian optimization.

* Revised the description of the "cutoff" argument of gbts().

* Changed R (>= 3.3.1) in the DESCRIPTION file to R (>= 3.3.0) to address the 
installation error from OS X on CRAN, which uses R version 3.3.0 at the time.

# gbts 1.0.0

* This is the initial release.



