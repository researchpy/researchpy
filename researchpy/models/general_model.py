import numpy as np
from researchpy.models.core_model import CoreModel

#from researchpy.objective_functions import likelihood
from researchpy.objective_functions.likelihood import neg_log_likelihood, gradient_neg_log_likelihood
from researchpy.optimize.iterative_algorithms import scipy_minimize, newton_raphson
from researchpy.models.postestimation import LikelihoodRatioTest

from scipy.stats import chi2



class GeneralModel(CoreModel):
    """

    This is a subclass of core_model for generalized statistical models such as logistic, poisson, and etc.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="mle",
                 obj_function="log-likelihood",
                 solver_options=None):

        self.__name__ = "Researchpy.GeneralModel"

        if data is None:
            data = {}
        if solver_options is None:
            solver_options = {"algorithm": "newton-raphson", "tol": 1e-7, "max_iter": 300, "display": True}


        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_options=solver_options, solver_method=solver_method, obj_function=obj_function)



    def __initialize_betas(self, initial_betas=None, initial_betas_method=None):

        if initial_betas is not None and initial_betas_method is not None:
            return("Warning: Both initial_betas and initial_betas_method were provided. Ignoring initial_betas_method and using provided initial_betas.")

        if initial_betas is not None:
            if isinstance(initial_betas, np.ndarray) and initial_betas.shape == (self.k, 1):
                self.model_data["betas"] = initial_betas
            else:
                raise ValueError(f"initial_betas must be a numpy array of shape ({self.k}, 1), but got {initial_betas.shape}")

        elif initial_betas_method.lower() == "ols":
            self._CoreModel__ols_fit()

        # Default initialization based on model type
        else:
            if self.__name__ in ["researchpy.Logistic", "researchpy.Logit"]:
                """Initialize betas with smarter starting values."""
                # Start with zeros (better than OLS for binary outcomes)
                betas = np.ones((self.n, self.k))

                # Set intercept to log odds of outcome proportion
                y_mean = np.mean(self.DV)
                if 0 < y_mean < 1:
                    betas[0] = np.log(y_mean / (1 - y_mean))

                self.model_data["betas"] = betas.reshape(-1, 1)



    def _neg_log_likelihood(self, params, *args, **kwargs):

        return neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.solver_options,
            distribution_family=self._family,
            link_function=self._link,
            tracker=self._OptimizationTracker
        )

    def _gradient_neg_log_likelihood(self, params):
        """Gradient of negative log-likelihood."""
        return gradient_neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.solver_options,
            distribution_family=self._family,
            link_function=self._link
        )


    def __fit_model(self):
        """Fit the model using scipy.optimize with fallback."""

        self.logL = []
        self.nfev = -1
        converged = False

        # Try scipy.optimize first
        try:
            if self.solver_options["display"]:
                print(f"Starting optimization with {self.solver_options['algorithm']}...\n")

            # Fit the full model
            result = scipy_minimize(
                fun=self._neg_log_likelihood,
                x0=self.model_data["betas"].flatten(),
                jac=self._gradient_neg_log_likelihood,
                method=self.solver_options["algorithm"],
                options={
                    'maxiter': self.solver_options["max_iter"],
                    'gtol': self.solver_options["tol"]
                },
                callback=None
            )

            if result.success:
                self.model_data["betas"] = result.x.reshape(-1, 1)
                self.logL.append(-result.fun)
                self.nfev = result.nfev
                converged = True

                # Perform Likelihood Ratio Test (full model vs null)
                self._lr_test = LikelihoodRatioTest(self, store_null=True)
                self.LL_null = self._lr_test.LL_restricted
                self.LR_chi2 = self._lr_test.LR_chi2
                self.model_df = self._lr_test.df
                self.model_p_value = self._lr_test.p_value

                if self.solver_options["display"]:
                    print(f"\nLR Chi^2: {self.LR_chi2:.4f}, df: {self.model_df}, p-value: {self.model_p_value:.4g}")
                    print("")
                    print(f"Optimization converged in {result.nfev} iterations")
                    print(f"Final log-likelihood: {-result.fun:.4f}")
            else:
                if self.solver_options["display"]:
                    print(f"Warning: scipy optimization did not converge ({result.message})")
                    print("Falling back to Newton-Raphson...")

        except Exception as e:
            if self.solver_options["display"]:
                print(f"scipy.optimize failed: {e}")
                print("Falling back to Newton-Raphson...")

        # Fallback to Newton-Raphson if scipy failed
        if not converged:
            self.model_data["betas"], self.logL = newton_raphson(
                IV=self.IV,
                DV=self.DV,
                betas=self.model_data["betas"],
                tol=self.solver_options["tol"],
                max_iter=self.solver_options["max_iter"],
                display=self.solver_options["display"]
            )

            # Perform Likelihood Ratio Test for Newton-Raphson path
            self.nfev = len(self.logL)
            self._lr_test = LikelihoodRatioTest(self, store_null=True)
            self.LL_null = self._lr_test.LL_restricted
            self.LR_chi2 = self._lr_test.LR_chi2
            self.model_df = self._lr_test.df
            self.model_p_value = self._lr_test.p_value

            if self.solver_options["display"]:
                print(f"\nLR Chi^2: {self.LR_chi2:.4f}, df: {self.model_df}, p-value: {self.model_p_value:.4g}")



    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def objective_function(self):
        if self.obj_func.lower() in ["log-likelihood", "log likelihood", "ll"]:
            y_e = predict(estimate="predict_y")
            objective = likelihood.log_likelihood(y_e)

        return objective


    def _regression_base_table(self):

        self._CoreModel__regression_base_table()


    def _table_regression_results(self, report=None, return_type="Dataframe",
                                  decimals=None,
                                 pretty_format=True, *args):
        
        if decimals is None:
            decimals = {"Coef.": 2, "Std. Err.": 4, "test_stat": 2, "test_stat_p": 4,
                        "CI": 2, "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4,
                        "Sum of Squares": 4, 'Degrees of Freedom': 1,
                        'Mean Squares': 4, 'Effect size': 4}

        self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

        if return_type == "Dataframe":
            return self._CoreModel__regression_base_table()

        elif return_type == "Dictionary":
            return self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")