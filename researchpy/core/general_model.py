import numpy as np
from researchpy.models.base import CoreModel
from researchpy.core.containerclasses import SolverOptions

from researchpy.objective_functions.likelihood import neg_log_likelihood, gradient_neg_log_likelihood
from researchpy.optimize.iterative_algorithms import scipy_minimize, newton_raphson
from researchpy.models.postestimation import LikelihoodRatioTest

from scipy.stats import chi2



class GeneralModel(CoreModel):
    """

    This is a subclass of core_model for generalized statistical models such as logistic, poisson, and etc.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal", solver_options=None,
                 table_decimals=None):

        self.__name__ = "Researchpy.GeneralModel"

        if data is None:
            data = {}

        # Build SolverOptions: start with GeneralModel defaults, then overlay user input.
        base_defaults = SolverOptions(
            method="mle",
            algorithm="newton-raphson",
            obj_function="log-likelihood",
            tol=1e-7,
            max_iter=300,
            display=True
        )

        if solver_options is None:
            resolved_solver_options = base_defaults
        elif isinstance(solver_options, SolverOptions):
            resolved_solver_options = solver_options
        else:
            # User passed a dict — override base defaults with user values
            resolved_solver_options = base_defaults.with_overrides(solver_options)


        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_options=resolved_solver_options,
                         table_decimals=table_decimals)



    def __initialize_betas(self, initial_betas=None, initial_betas_method=None):

        if initial_betas is not None and initial_betas_method is not None:
            return("Warning: Both initial_betas and initial_betas_method were provided. Ignoring initial_betas_method and using provided initial_betas.")

        if initial_betas is not None:
            if isinstance(initial_betas, np.ndarray) and initial_betas.shape == (self.k, 1):
                self.CoefResults.betas = initial_betas
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

                self.CoefResults.betas = betas.reshape(-1, 1)



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
            if self.solver_options.display:
                print(f"Starting optimization with {self.solver_options.algorithm}...\n")

            # Fit the full model
            result = scipy_minimize(
                fun=self._neg_log_likelihood,
                x0=self.CoefResults.betas.flatten(),
                jac=self._gradient_neg_log_likelihood,
                method=self.solver_options.algorithm,
                options=self.solver_options.to_scipy_options(),
                callback=None
            )

            if result.success:
                self.CoefResults.betas = result.x.reshape(-1, 1)
                self.logL.append(-result.fun)
                self.nfev = result.nfev
                converged = True

                # Perform Likelihood Ratio Test (full model vs null)
                self._lr_test = LikelihoodRatioTest(self, store_null=True)
                self.LL_null = self._lr_test.LL_restricted
                self.LR_chi2 = self._lr_test.LR_chi2
                self.model_df = self._lr_test.df
                self.model_p_value = self._lr_test.p_value

                if self.solver_options.display:
                    print(f"\nLR Chi^2: {self.LR_chi2:.4f}, df: {self.model_df}, p-value: {self.model_p_value:.4g}")
                    print("")
                    print(f"Optimization converged in {result.nfev} iterations")
                    print(f"Final log-likelihood: {-result.fun:.4f}")
            else:
                if self.solver_options.display:
                    print(f"Warning: scipy optimization did not converge ({result.message})")
                    print("Falling back to Newton-Raphson...")

        except Exception as e:
            if self.solver_options.display:
                print(f"scipy.optimize failed: {e}")
                print("Falling back to Newton-Raphson...")

        # Fallback to Newton-Raphson if scipy failed
        if not converged:
            betas, self.logL = newton_raphson(
                IV=self.IV,
                DV=self.DV,
                betas=self.CoefResults.betas,
                tol=self.solver_options.tol,
                max_iter=self.solver_options.max_iter,
                display=self.solver_options.display
            )
            self.CoefResults.betas = betas

            # Perform Likelihood Ratio Test for Newton-Raphson path
            self.nfev = len(self.logL)
            self._lr_test = LikelihoodRatioTest(self, store_null=True)
            self.LL_null = self._lr_test.LL_restricted
            self.LR_chi2 = self._lr_test.LR_chi2
            self.model_df = self._lr_test.df
            self.model_p_value = self._lr_test.p_value

            if self.solver_options.display:
                print(f"\nLR Chi^2: {self.LR_chi2:.4f}, df: {self.model_df}, p-value: {self.model_p_value:.4g}")



    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the summary header for generalized models.

        Shows model name and log-likelihood value.

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        lines = [self._get_model_display_name()]

        if hasattr(self, 'logL') and self.logL:
            ll_val = self.logL[-1] if isinstance(self.logL, list) else self.logL
            if ll_val is not None:
                lines.append(f"Log likelihood = {ll_val:.4f}")

        return lines


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the summary header for generalized models.

        When *descriptives_df* is provided the values are read from that
        DataFrame via ``to_string(header=False)`` so the summary is driven
        entirely by the DataFrames returned from ``self.results()``.

        Otherwise falls back to rendering from ``self`` attributes directly.

        Parameters
        ----------
        width : int
            Available character width.
        descriptives_df : DataFrame or None
            Descriptives DataFrame from ``self.results()`` (index-oriented:
            stat names as index, values in column 0).

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        if descriptives_df is not None:
            return descriptives_df.to_string(header=False).split("\n")

        # Fallback: build from self attributes
        lines = [f"Number of obs = {self.n:>8}"]

        if hasattr(self, 'LR_chi2'):
            df = getattr(self, 'model_df', '?')
            lines.append(f"LR chi2({df})    = {self.LR_chi2:>8.4f}")

        if hasattr(self, 'model_p_value'):
            lines.append(f"Prob > chi2   = {self.model_p_value:>8.4f}")

        if hasattr(self, 'nfev') and self.nfev is not None:
            lines.append(f"N iterations  = {self.nfev:>8}")

        return lines


    def objective_function(self):
        if self.obj_func.lower() in ["log-likelihood", "log likelihood", "ll"]:
            y_e = predict(estimate="predict_y")
            objective = likelihood.log_likelihood(y_e)

        return objective


    def _regression_base_table(self):

        self._CoreModel__regression_base_table()


    def _table_regression_results(self, report=None, return_type="Dataframe",
                                  table_decimals=None, pretty_format=True, *args):

        self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, table_decimals=table_decimals)

        if return_type == "Dataframe":
            return self._CoreModel__regression_base_table()

        elif return_type == "Dictionary":
            return self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")



    def __table_regression_results(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Build the coefficient table using parent's private method (returns Pandas DataFrame)
        return self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, table_decimals=self._table_decimals)




    def _get_results(self, report_as=None, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):
        """
        Return the regression results.

        Parameters
        ----------
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        tuple
            If return_type is "Dataframe": (descriptives_df, model_results_df, coefficients_df)
            If return_type is "Dictionary": (descriptives_dict, model_results_dict, coefficients_dict)
        """
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # Build the coefficient table
        result = self.__table_regression_results(return_type=return_type,
                                        pretty_format=pretty_format,
                                        table_decimals=self._table_decimals)


        if return_type.lower() in ["dataframe", "df", "pandas.dataframe"]:
            return result if result is not None else self.regression_table_info

        elif return_type.lower() in ["dictionary", "dict"]:
            return self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")