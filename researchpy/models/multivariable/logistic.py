from scipy.stats import norm
from scipy.special import expit

from researchpy.models.general_model import GeneralModel
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
    solver_options : dict, optional
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

    def __init__(self, formula_like, data=None,
                 solver_method="mle", solver_options=None,
                 initial_betas=None, initial_betas_method="ols"):

        if data is None:
            data = {}

        if solver_options is None:
            solver_options = {}

        base_solver_options = {"algorithm": "BFGS", # 'BFGS' (Broyden-Fletcher-Goldfarb-Shanno algorithm, 'nr' (Newton-Raphson)
                               "tol": 1e-7,
                               "max_iter": 1000,
                               "display": True,
                               "regularization": None,  # None, 'l1', or 'l2'
                               "alpha": 0.0  # Regularization strength
                               }
        solver_options = base_solver_options | solver_options

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.__name__ = "Researchpy.LogisticRegression"

        self.model_data = {}


        # Initializing betas
        self._GeneralModel__initialize_betas(initial_betas=initial_betas, initial_betas_method=initial_betas_method)

        # Fit the model
        self._GeneralModel__fit_model()

        # Compute standard errors and statistics
        self._compute_statistics()



    def _compute_statistics(self):
        """Compute standard errors, test statistics, and p-values."""
        linear_pred = self.IV @ self.model_data["betas"]
        p = expit(linear_pred)
        w = p * (1 - p).reshape(-1, 1)
        X_w = self.IV * w

        # Covariance matrix (inverse of Fisher information)
        try:
            cov_matrix = np.linalg.inv(self.IV.T @ X_w)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(self.IV.T @ X_w)

        self.model_data["standard_errors"] = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)

        # Wald z-statistics and p-values
        self.model_data["test_stat"] = self.model_data["betas"] / self.model_data["standard_errors"]
        self.model_data["test_stat_p_values"] = 2 * norm.sf(np.abs(self.model_data["test_stat"]))

        # Compute confidence intervals
        self._CoreModel__compute_confidence_intervals(add_to_model_data=True)



    def predict(self, estimate=None):
        """Predict probabilities."""
        linear_pred = self.IV @ self.model_data["betas"]
        return expit(linear_pred)



    def results(self, report="or", return_type="Dataframe", pretty_format=True,
                decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 2, "test_stat_p": 4, "CI": 2,
                          "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                          'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                *args):

        model_meta = {"Model": ["Logistic Regression"],
                      "Log likelihood = ": [self.logL],
                      "N iterations for optimization convergence = ": [self.nfev]}

        model_description = {"Number of observations = ": [self.nobs],
                             f"LR Chi^2({self.model_df}) = ": [self.LR_chi2],
                             "Prob > Chi^2 = ": [self.model_p_value]}

        if report.lower() in ["or", "odds ratio", "odds_ratio"]:
            if self._beta_type == "coef":
                self.model_data["betas"] = np.exp(self.model_data["betas"])
                self.model_data["conf_int_lower"] = np.exp(self.model_data["conf_int_lower"])
                self.model_data["conf_int_upper"] = np.exp(self.model_data["conf_int_upper"])
                self._beta_type = "odds ratio"


            if return_type == "Dataframe":
                table = self._table_regression_results(return_type=return_type, pretty_format=pretty_format,
                                                       decimals=decimals)

                model_meta = pd.DataFrame.from_dict(model_meta, orient='index')
                model_description = pd.DataFrame.from_dict(model_description, orient='index')

                if self._beta_type == "odds ratio":
                    return model_meta, model_description, table.rename(columns={"Coef.": "Odds Ratio"})
                else:
                    return model_meta, model_description, table

            else:
                dct = self._table_regression_results(return_type=return_type, pretty_format=pretty_format,
                                                     decimals=decimals)
                if self._beta_type == "odds ratio":
                    dct = {('Odds Ratio' if k == 'Coef.' else k): v for k, v in dct.items()}
                    return model_meta, model_description, dct
                else:
                    return model_meta, model_description, dct


    def _get_summary_parts(self, report="or"):
        """
        Return (descriptives_df, coef_df) for the Logistic Regression summary.

        Calls ``self.results()`` and unpacks the 3-tuple into the two
        DataFrames that ``CoreModel.summary()`` needs.

        Parameters
        ----------
        report : str, optional
            ``"or"`` for odds ratios (default), ``"coef"`` for raw coefficients.

        Returns
        -------
        tuple of (descriptives_df, coef_df)
        """
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            _model_meta_df, model_description_df, coef_df = self.results(
                report=report, return_type="Dataframe", pretty_format=True
            )
        return model_description_df, coef_df


    def summary(self, total_width=78, return_string=False, report="or", decimals=None):
        """
        Print a formatted summary of the logistic regression results.

        Parameters
        ----------
        total_width : int, optional
            Character width for the summary output. Default is 78.
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False.
        report : str, optional
            ``"or"`` for odds ratios (default), ``"coef"`` for raw
            log-odds coefficients.
        decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        str or None
            If *return_string* is True, returns the formatted summary string.
            Otherwise, prints to terminal and returns None.
        """
        # Store report preference so _get_summary_parts() can pick it up
        self._summary_report = report
        result = super().summary(total_width=total_width, return_string=return_string, decimals=decimals)
        del self._summary_report
        return result


    def _get_summary_parts(self, **kwargs):
        """
        Return (descriptives_df, coef_df) for the Logistic Regression summary.

        Calls ``self.results()`` and unpacks the 3-tuple into the two
        DataFrames that ``CoreModel.summary()`` needs.

        Returns
        -------
        tuple of (descriptives_df, coef_df)
        """
        report = getattr(self, '_summary_report', 'or')
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model_summary_df, model_description_df, coef_df = self.results(
                report=report, return_type="Dataframe", pretty_format=True
            )
        return model_summary_df, model_description_df, coef_df

# Convenience alias
Logistic = LogisticRegression
