import numpy as np
from scipy.special import expit

def log_likelihood(y_e):

    #return np.sum(y_e - np.sum(np.log((1 + y_e))))
    return np.sum(y_e) - np.sum(np.log((1 + y_e)))



class LikelihoodTracker:
    """Class to track the log-likelihood value."""
    def __init__(self):
        self.current_log_likelihood = float('-inf')  # Initialize to a valid value


def neg_log_likelihood(params, IV, DV, solver_options, distribution_family="binomial", link_function="logit", tracker=None):
    """Negative log-likelihood function for scipy.optimize."""
    params = params.reshape(-1, 1)  # Ensure params is a column vector
    linear_pred = IV @ params

    # Apply the link function
    if link_function == "logit":
        p = expit(linear_pred)  # Numerically stable sigmoid
    else:
        raise NotImplementedError(f"Link function '{link_function}' is not implemented.")

    # Clip to avoid log(0)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    # Compute log-likelihood based on the distribution family
    if distribution_family == "binomial":
        ll = -np.sum(DV * np.log(p) + (1 - DV) * np.log(1 - p))
    else:
        raise NotImplementedError(f"Distribution family '{distribution_family}' is not implemented.")

    # Add regularization if specified
    if solver_options.get("regularization") == "l2":
        alpha = solver_options.get("alpha", 0.0)
        # Don't regularize intercept (first coefficient)
        ll += alpha * np.sum(params[1:] ** 2)

    elif solver_options.get("regularization") == "l1":
        alpha = solver_options.get("alpha", 0.0)
        ll += alpha * np.sum(np.abs(params[1:]))

    # Store the log-likelihood value in the tracker if provided
    if tracker is not None:
        tracker.current_cost = ll
        tracker.current_cost_index += 1

        if solver_options["display"]:
            #print(f"{tracker.current_cost_index} Bernoulli Log-likelihood = {-ll:.4f}")
            print(f"{tracker.current_cost_index} Log-likelihood = {-ll:.4f}")

    return ll



def gradient_neg_log_likelihood(params, IV, DV, solver_options, distribution_family="binomial", link_function="logit"):
    """Gradient of negative log-likelihood."""
    params = params.reshape(-1, 1)  # Ensure params is a column vector
    linear_pred = IV @ params

    # Apply the link function
    if link_function == "logit":
        p = expit(linear_pred)
    else:
        raise NotImplementedError(f"Link function '{link_function}' is not implemented.")

    # Compute gradient based on the distribution family
    if distribution_family == "binomial":
        grad = -IV.T @ (DV - p)
    else:
        raise NotImplementedError(f"Distribution family '{distribution_family}' is not implemented.")

    # Add regularization gradient if specified
    if solver_options.get("regularization") == "l2":
        alpha = solver_options.get("alpha", 0.0)
        reg_grad = np.zeros_like(params)
        reg_grad[1:] = 2 * alpha * params[1:]  # Don't regularize intercept
        grad += reg_grad

    elif solver_options.get("regularization") == "l1":
        alpha = solver_options.get("alpha", 0.0)
        reg_grad = np.zeros_like(params)
        reg_grad[1:] = alpha * np.sign(params[1:])
        grad += reg_grad

    return grad.flatten()  # Return flattened gradient for scipy.optimize