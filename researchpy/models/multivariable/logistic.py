from scipy.stats import norm
from scipy.special import expit

from researchpy.models.general_model import GeneralModel
from researchpy.core.containerclasses import ModelResults, SolverOptions
from researchpy.utility import *
from researchpy.predict import predict


class LogisticRegression(GeneralModel):
    """
    Logistic regression for binary outcomes using Maximum Likelihood Estimation (MLE).

    This class provides logistic regression analysis with support for:
    - Multiple optimization algorithms (BFGS, Newton-Raphson)
    - L1 and L2 regularization
    - Odds ratio reporting
    - Confidence intervals
    - Model fit statistics (LR Chi-squared, etc.)

    Parameters
    ----------
    formula_like : str
        A string representing a valid Patsy formula (e.g., "outcome ~ predictor1 + predictor2").
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    solver_method : str, optional
        Method for fitting the model. Default is "mle".
    solver_options : dict or SolverOptions, optional
        Options for the solver including:
        - algorithm: "BFGS" (default) or "newton-raphson"
        - tol: Convergence tolerance (default 1e-7)
        - max_iter: Maximum iterations (default 1000)
        - display: Whether to print iteration info (default True)
        - regularization: None, "l1", or "l2"
        - alpha: Regularization strength (default 0.0)
    initial_betas : ndarray, optional
        Initial coefficient values.
    initial_betas_method : str, optional
        Method to initialize betas. Default is "ols".

    Examples
    --------
    >>> import researchpy as rp
    >>> model = rp.LogisticRegression("outcome ~ age + treatment", data=df)
    >>> model.results()

    See Also
    --------
    Logistic : Alias for LogisticRegression
    """

    def __init__(self, formula_like, data=None, display_summary=True,
                 solver_options=None,
                 initial_betas=None, initial_betas_method="ols"):

        if data is None:
            data = {}

        # Build SolverOptions: start with LogisticRegression-specific defaults, then overlay user input.
        base_defaults = SolverOptions(
            method="mle",
            algorithm="BFGS",
            obj_function="log-likelihood",
            tol=1e-7,
            max_iter=1000,
            display=True,
            regularization=None,
            alpha=0.0
        )

        if solver_options is None:
            resolved_solver_options = base_defaults
        elif isinstance(solver_options, SolverOptions):
            resolved_solver_options = solver_options
        else:
            # User passed a dict — override base defaults with user values
            resolved_solver_options = base_defaults.with_overrides(solver_options)

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_options=resolved_solver_options,
                         family="binomial", link="logit")

        self.__name__ = "Researchpy.LogisticRegression"



        # Initializing betas
        self._GeneralModel__initialize_betas(initial_betas=initial_betas, initial_betas_method=initial_betas_method)

        # Fit the model
        self._GeneralModel__fit_model()

        # Compute standard errors and statistics
        self._compute_statistics()

        # Build ModelResults (results() sets self.ModelResults internally)
        self.results(report_as="or", return_type="Dataframe", pretty_format=True)

        # Display the model results summary
        if display_summary:
            self.summary()



    def _compute_statistics(self):
        """Compute standard errors, test statistics, and p-values."""
        linear_pred = self.IV @ self.CoefResults.betas
        p = expit(linear_pred)
        w = p * (1 - p).reshape(-1, 1)
        X_w = self.IV * w

        # Covariance matrix (inverse of Fisher information)
        try:
            cov_matrix = np.linalg.inv(self.IV.T @ X_w)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(self.IV.T @ X_w)

        self.CoefResults.std_error = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)

        # Wald z-statistics and p-values
        self.CoefResults.test_stat = self.CoefResults.betas / self.CoefResults.std_error
        self.CoefResults.p_value = 2 * norm.sf(np.abs(self.CoefResults.test_stat))

        # Compute confidence intervals
        self._CoreModel__compute_confidence_intervals()



    def predict(self, estimate=None):
        """Predict probabilities."""
        linear_pred = self.IV @ self.CoefResults.betas
        return expit(linear_pred)




    def results(self, report_as="or", return_type="Dataframe", pretty_format=True,
                table_decimals=None, *args):
        """
        Return the logistic regression results as a ``ModelResults`` dataclass.

        Parameters
        ----------
        report_as : str, optional
            ``"or"`` for odds ratios (default), ``"coef"`` for raw log-odds.
        return_type : str, optional
            ``"Dataframe"`` (default) or ``"Dictionary"``.
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        ModelResults
            A dataclass with fields:
            - ``model_name``: ``"Logistic Regression"``
            - ``fit_statistics``: Combined fit statistics (DataFrame or dict)
            - ``model_table``: ``None`` (MLE-based model, no SS decomposition)
            - ``coefficients``: Coefficient / odds ratio table (DataFrame or dict)
            - ``details``: ``None``

            Supports tuple unpacking::

                name, fit_stats, model_table, coefs, details = model.results()

            Or attribute access::

                result = model.results()
                result.fit_statistics
                result.coefficients
        """
        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Need to convert to odds ratio before calling _get_results()
        if report_as.lower() in ["or", "odds ratio", "odds_ratio"]:
            if self._beta_type == "coef":
                self.CoefResults.betas = np.exp(self.CoefResults.betas)
                self.CoefResults.conf_int_lower = np.exp(self.CoefResults.conf_int_lower)
                self.CoefResults.conf_int_upper = np.exp(self.CoefResults.conf_int_upper)
                self._beta_type = "odds ratio"

        # Combined fit statistics dictionary
        ll_val = self.logL[-1] if isinstance(self.logL, list) else self.logL
        fit_statistics_dict = {
            "Number of observations = ": [self.n],
            f"LR Chi^2({self.model_df}) = ": [round(float(self.LR_chi2), 4)],
            "Prob > Chi^2 = ": [round(float(self.model_p_value), 4)],
            "Log likelihood = ": [round(float(ll_val), 4) if ll_val is not None else None],
            "N iterations = ": [self.nfev],
        }

        # Returns as pd.DataFrame by default
        raw = self._get_results(return_type=return_type, pretty_format=pretty_format,
                                table_decimals=self._table_decimals)

        # Build coefficient table
        if return_type == "Dataframe":
            fit_stats_out = pd.DataFrame.from_dict(fit_statistics_dict, orient='index')

            if self._beta_type == "odds ratio":
                raw = raw.rename(columns={"Coef.": "Odds Ratio"})

            self.ModelResults = ModelResults(
                model_name=self._get_model_display_name(),
                fit_statistics=fit_stats_out,
                model_table=None,
                coefficients=raw,
            )
            return self.ModelResults

        elif return_type == "Dictionary":
            if self._beta_type == "odds ratio":
                raw = {('Odds Ratio' if k == 'Coef.' else k): v for k, v in raw.items()}

            self.ModelResults = ModelResults(
                model_name=self._get_model_display_name(),
                fit_statistics=fit_statistics_dict,
                model_table=None,
                coefficients=raw,
            )
            return self.ModelResults

        else:
            raise ValueError(
                "Not a valid return type option, please use either "
                "'Dataframe' or 'Dictionary'."
            )



    def summary(self, total_width=78, return_string=False, report_as="or", table_decimals=None):
        """
        Print a formatted summary of the logistic regression results.

        Parameters
        ----------
        total_width : int, optional
            Character width for the summary output. Default is 78.
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False.
        report_as : str, optional
            ``"or"`` for odds ratios (default), ``"coef"`` for raw
            log-odds coefficients.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        str or None
            If *return_string* is True, returns the formatted summary string.
            Otherwise, prints to terminal and returns None.
        """
        # Ensure ModelResults is up-to-date with the requested report format
        self.results(report_as=report_as, return_type="Dataframe", pretty_format=True,
                     table_decimals=table_decimals)

        return super().summary(total_width=total_width, return_string=return_string,
                               table_decimals=table_decimals)


# Convenience alias
Logistic = LogisticRegression
