// ///////////////////////////////////////////////////////////////////////////////////////
//
// RUN A SIMPLE RANDOM INTERCEPTS MODEL
//
// ///////////////////////////////////////////////////////////////////////////////////////

#include <TMB.hpp>
using namespace density;

// Read in an R list specifying a prior
// List should contain three objects: "name", "par1", "par2"
template<class Type>
struct prior_type {
  std::string name;
  Type par1;
  Type par2;

  prior_type(SEXP x){
    name = CHAR(STRING_ELT(getListElement(x,"name"), 0));
    par1 = asVector<float>(getListElement(x,"par1"))[0];
    par2 = asVector<float>(getListElement(x,"par2"))[0];
  }
};

// evaluate a prior for an object
template<class Type>
Type evaluate_prior_density(prior_type<Type> prior, Type param, bool log_density = true){
  Type density;

  if(prior.name == "gaussian"){
    density = dnorm(param, prior.par1, prior.par2, log_density);
  } else if(prior.name == "gamma"){
    density = dgamma(param, prior.par1, prior.par2, log_density);
  } else if(prior.name == "beta"){
    density = dbeta(param, prior.par1, prior.par2, log_density);
  } else {
    // Prior name must be one of "gaussian", "gamma", or "beta"
    exit(1);
  }

  return density;
}

// TMB objective function
template<class Type>
Type objective_function<Type>::operator() () {

  // Model inputs (passed from R)
  // Model data
  DATA_VECTOR(y_i); // Outcomes (vector of real numbers)
  DATA_VECTOR(x_i); // Single covariate (vector of real numbers)
  DATA_IVECTOR(group_i); // Group identifier for each observation (vector of integers)
  // Model priors
  DATA_STRUCT(prior_alpha, prior_type); // Prior on alpha (intercept)
  DATA_STRUCT(prior_beta, prior_type); // Prior on beta (covariate effect)
  DATA_STRUCT(prior_sigma_group, prior_type); // Prior on sigma_group
  DATA_STRUCT(prior_sigma_error, prior_type); // Prior on sigma_epsilon
  // Model parameters
  PARAMETER(alpha);
  PARAMETER(beta);
  PARAMETER(log_sigma_group); // Transformed into sigma_group
  PARAMETER(log_sigma_error); // Transformed into sigma_error
  PARAMETER_VECTOR(group_res); // Length = number of groups

  // Transform some parameters into measurement space
  Type sigma_group = exp(log_sigma_group);
  Type sigma_error = exp(log_sigma_error);

  // Instantiate joint negative log likelihood (JNLL)
  Type jnll = 0;

  // Evaluate priors
  jnll -= evaluate_prior_density(prior_alpha, alpha);
  jnll -= evaluate_prior_density(prior_beta, beta);
  jnll -= evaluate_prior_density(prior_sigma_group, sigma_group);
  jnll -= dnorm(group_res, Type(0.0), sigma_group, true).sum();
  jnll -= evaluate_prior_density(prior_sigma_error, sigma_error);

  // Evaluate likelihood of parameters given data
  vector<Type> estimates = alpha + beta * x_i + group_res(group_i);
  jnll -= dnorm(y_i, estimates, sigma_error, true).sum();

  // Return JNLL
  return jnll;
}
