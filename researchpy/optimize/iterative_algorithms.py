import numpy as np
from scipy.special import expit
from scipy.optimize import minimize



def scipy_minimize(fun, x0, jac, method, options):
    """Wrapper for scipy.optimize.minimize."""
    return minimize(fun=fun, x0=x0, jac=jac, method=method, options=options)



def newton_raphson(IV, DV, betas, tol, max_iter, display):
    """Newton-Raphson optimization algorithm."""
    it = 0
    error = np.ones_like(betas)

    logL = []

    while np.any(error > tol) and it < max_iter:
        linear_pred = IV @ betas
        p = expit(linear_pred)  # Use expit for stability

        w = p * (1 - p).reshape(-1, 1)
        H = -(IV.T @ (IV * w))
        G = IV.T @ (DV - p)

        try:
            betas_new = betas - np.linalg.inv(H) @ G
        except np.linalg.LinAlgError:
            betas_new = betas - np.linalg.pinv(H) @ G

        error = np.abs(betas_new - betas)
        betas = betas_new

        ll = np.sum(DV * np.log(p + 1e-12) + (1 - DV) * np.log(1 - p + 1e-12))
        logL.append(ll)

        it += 1
        if display:
            print(f"NR Iteration {it}: Log-likelihood = {ll:.4f}")

    if display:
        print(f"Newton-Raphson completed in {it} iterations")

    return betas, logL
