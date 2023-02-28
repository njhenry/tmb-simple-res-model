#' Simulate multivariate normal draws from a mean vector and precision matrix
#'
#' @details Sample multivariate normal draws from a mean vector and a joint precision
#'   matrix. This function uses the matrix square root of the precision matrix, created
#'   using the Cholesky decomposition. For more information, see
#'   \href{https://bit.ly/3AYqLOt}{this tutorial on multivariate normal sampling}.
#'
#' @param mu Vector of parameter means
#' @param prec Joint precision matrix. Should be a matrix with rows and columns in the
#'   same ordering as `mu`
#' @param num_draws number of draws
#'
#' @return Matrix of parameter draws with dimensions (# parameters) by (# draws)
#'
generate_draws <- function(mu, prec, num_draws){
  require("Matrix")
  # Check inputs
  if(nrow(prec) != ncol(prec)) stop("Precision matrix must be square.")
  if(length(mu) != nrow(prec)){
    stop("Length of mean vector 'mu' must match the dimensions of the precision matrix.")
  }
  # Random samples from N(0, 1)
  z_matrix <- matrix(rnorm(length(mu) * num_draws), ncol=num_draws)
  # Cholesky lower triangular square root of the precision matrix
  L_inv <- Matrix::Cholesky(prec)
  # Solve for the noise matrix to get draws
  mvn_draws <- as.matrix(
    solve(
      as(L_inv, "pMatrix"),
      solve(t(as(L_inv, "Matrix")), z_matrix)
    )
  )
  return(mu + mvn_draws)
}
