import numpy as np

def log_likelihood(y_e):

    #return np.sum(y_e - np.sum(np.log((1 + y_e))))
    return np.sum(y_e) - np.sum(np.log((1 + y_e)))






def neg_log_likelihood(self, params):
    """Negative log-likelihood function for scipy.optimize."""
    params = params.reshape(-1, 1)  # Ensure params is a column vector
    linear_pred = self.IV @ params
    p = expit(linear_pred)  # Numerically stable sigmoid

    # Clip to avoid log(0)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    ll = -np.sum(self.DV * np.log(p) + (1 - self.DV) * np.log(1 - p))

    # Add regularization if specified
    if self.solver_options.get("regularization") == "l2":
        alpha = self.solver_options.get("alpha", 0.0)
        # Don't regularize intercept (first coefficient)
        ll += alpha * np.sum(params[1:] ** 2)

    elif self.solver_options.get("regularization") == "l1":
        alpha = self.solver_options.get("alpha", 0.0)
        ll += alpha * np.sum(np.abs(params[1:]))

    return ll



def gradient_neg_log_likelihood(self, params):
    """Gradient of negative log-likelihood."""
    params = params.reshape(-1, 1)  # Ensure params is a column vector
    linear_pred = self.IV @ params
    p = expit(linear_pred)

    grad = -self.IV.T @ (self.DV - p)

    # Add regularization gradient if specified
    if self.solver_options.get("regularization") == "l2":
        alpha = self.solver_options.get("alpha", 0.0)
        reg_grad = np.zeros_like(params)
        reg_grad[1:] = 2 * alpha * params[1:]  # Don't regularize intercept
        grad += reg_grad

    elif self.solver_options.get("regularization") == "l1":
        alpha = self.solver_options.get("alpha", 0.0)
        reg_grad = np.zeros_like(params)
        reg_grad[1:] = alpha * np.sign(params[1:])
        grad += reg_grad

    return grad.flatten()  # Return flattened gradient for scipy.optimize