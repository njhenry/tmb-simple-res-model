## #######################################################################################
##
## RUN SIMPLE RANDOM INTERCEPTS MODEL
##
## TRAINING: Using priors - Feb 24 2023
##
## #######################################################################################

library(TMB); library(data.table)

if(interactive()){
  # Change this if running in RStudio
  repo_dir <- '~/repos/tmb-simple-res-model'
  setwd(repo_dir)
}
source('draw_generation_helper_function.R')

# True values
n_obs <- 1e5
n_groups <- 100
true_alpha <- 3
true_beta <- 2
true_sigma_group <- .5
true_sigma_error <- 1

# Generate data
set.seed(20230227)
x_i <- runif(n = n_obs, min = -100, max = 100)
group_vals <- rnorm(n = n_groups, mean = 0, sd = true_sigma_group)
group_id <- rep(1:n_groups, length = n_obs)
group_id_zero_indexed <- group_id - 1
error_vals <- rnorm(n = n_obs, mean = 0, sd = true_sigma_error)
y_i <- true_alpha + true_beta * x_i + group_vals[group_id] + error_vals

# Define priors
prior_alpha <- list(name = 'gaussian', par1 = 0, par2 = 1000)
prior_beta <- list(name = 'gaussian', par1 = 0, par2 = 1000)
prior_sigma_group <- list(name = 'gamma', par1 = 1, par2 = 1000)
prior_sigma_error <- list(name = 'gamma', par1 = 1, par2 = 1000)

# Prepare TMB inputs
tmb_data_list <- list(
  y_i = y_i,
  x_i = x_i,
  group_i = group_id_zero_indexed,
  prior_alpha = prior_alpha,
  prior_beta = prior_beta,
  prior_sigma_group = prior_sigma_group,
  prior_sigma_error = prior_sigma_error
)
tmb_parameter_list <- list(
  alpha = 0,
  beta = 0,
  log_sigma_group = 0,
  log_sigma_error = 0,
  group_res = rep(0, times = n_groups)
)

# Compile and load model
TMB::compile(here::here("random_intercept_model.cpp"))
if(Sys.info()[1] == 'Windows'){
  dyn.load("random_intercept_model")
} else {
  dyn.load("random_intercept_model.so")
}
# Create TMB model function
adfunction <- TMB::MakeADFun(
  data = tmb_data_list,
  parameters = tmb_parameter_list,
  DLL = "random_intercept_model",
  random = "group_res"
)

# Optimize model
optimized <- stats::nlminb(
  start = adfunction$par,
  objective = adfunction$fn,
  gradient = adfunction$gr,
  control = list(trace = TRUE)
)

# Get joint precision matrix
sdrep <- TMB::sdreport(adfunction, bias.correct = TRUE, getJointPrecision = TRUE)

# Generate parameter draws and summaries
mu <- c(sdrep$par.fixed, sdrep$par.random)
parameter_draws <- generate_draws(mu = mu, prec = sdrep$jointPrecision, num_draws = 250)
# log_sigmas => sigmas
parameter_draws[c(3,4), ] <- exp(parameter_draws[c(3,4), ])
param_names <- gsub('^log_', '', names(mu))
# Summarize draws
parameter_summaries <- data.table::data.table(
  param_names = param_names,
  mean = rowMeans(parameter_draws),
  lower = matrixStats::rowQuantiles(parameter_draws, probs = 0.025),
  upper = matrixStats::rowQuantiles(parameter_draws, probs = 0.975),
  simulated = c(true_alpha, true_beta, true_sigma_group, true_sigma_error, group_vals)
)
