import patsy
import pandas as pd

from researchpy.models.base import CoreModel
from researchpy.utility import *


class LinearModel(CoreModel):
    """

    This is a subclass of core_model for linear statistical models that use ordinary least squares.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95, display_summary=True,
                 family="gaussian", link="normal", solver_method="ols", obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True},
                 table_decimals=None):

        self.__name__ = "Researchpy.LinearModel"

        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_method=solver_method, obj_function=obj_function,
                         table_decimals=table_decimals)

        # Creating the J matrix
        J = self._j_matrix(add_to_model_data=False)

        # Creating the identify matrix
        I = self._identity_matrix(add_to_model_data=False)

        # Calculation of the Hat matrix
        self._hat_matrix(add_to_model_data=True)

        # OLS fit to compute the coefficients (betas)
        self._CoreModel__ols_fit(add_to_model_data=True)

        # Compute the model sum of square, degrees of freedom, mean squares, F-value, p-value, and effect size measures
        self.__model_sum_of_square_stats(add_to_model_data=True)

        # Compute standard errors and confidence intervals
        self.__compute_beta_se_and_stats(add_to_model_data=True)



    # Compute standard errors and statistics
    def __compute_beta_se_and_stats(self, add_to_model_data=True):
        variance_covariance_beta_matrix = self.__variance_covariance_beta_matrix(method="standard", to_return=True, add_to_self=False, add_to_model_data=False)

        ## Standard Errors
        self.model_data["standard_errors"] = (np.array(np.sqrt(variance_covariance_beta_matrix.diagonal()))).T

        ## T-statistics
        self.model_data["test_stat"] = self.model_data["betas"] * (1 / self.model_data["standard_errors"])

        ## Two-sided p-value
        self.model_data["test_stat_p_values"] = np.array([
            float((scipy.stats.t.sf(np.abs(t), self.model_data["degrees_of_freedom_residual"]) * 2).item())
            for t in self.model_data["test_stat"]
        ])

        self._CoreModel__compute_confidence_intervals(add_to_model_data=True)


    def __variance_covariance_residual_matrix(self, method="standard", to_return=True, add_to_self=False, add_to_model_data=False):

        if method == "standard":
            variance_covariance_residual_matrix = np.matrix(self.model_data["mse"] * (self.model_data["I"] - self.model_data["H"]))

        if add_to_self:
            self.variance_covariance_residual_matrix = variance_covariance_residual_matrix

        if add_to_model_data:
            self.model_data["variance_covariance_residual_matrix"] = variance_covariance_residual_matrix

        if to_return:
            return variance_covariance_residual_matrix


    def __variance_covariance_beta_matrix(self, method="standard", to_return=True, add_to_self=False, add_to_model_data=False):

        if method == "standard":
            try:
                variance_covariance_beta_matrix = np.matrix(self.model_data["mse"] * np.linalg.inv(self.IV.T @ self.IV))
            except np.linalg.LinAlgError:
                variance_covariance_beta_matrix = np.matrix(self.model_data["mse"] * np.linalg.pinv(self.IV.T @ self.IV))

        if add_to_self:
            self.variance_covariance_beta_matrix = variance_covariance_beta_matrix

        if add_to_model_data:
            self.model_data["variance_covariance_beta_matrix"] = variance_covariance_beta_matrix

        if to_return:
            return variance_covariance_beta_matrix




    def __model_sum_of_square_stats(self, to_return=False, add_to_self=False, add_to_model_data=True):

        predicted_y = self.IV @ self.model_data["betas"]    # predicted y values
        residuals = self.DV - predicted_y                   # Calculation of residuals (error)

        J = self._j_matrix(add_to_model_data=False)         # Creating the J matrix
        I = self._identity_matrix(add_to_model_data=False)  # Creating the identify matrix


        ### Sum of Squares
        # Total sum of squares (SSTO)
        self.model_data["sum_of_square_total"] = float((self.DV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Model sum of squares (SSR)
        self.model_data["sum_of_square_model"] = float((self.model_data["betas"].T @ self.IV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Error sum of squares (SSE)
        self.model_data["sum_of_square_residual"] = float((residuals.T @ residuals).item())

        ### Degrees of freedom
        # Model
        self.model_data["degrees_of_freedom_model"] = np.linalg.matrix_rank(self.IV) - 1

        # Error
        self.model_data["degrees_of_freedom_residual"] = self.n - np.linalg.matrix_rank(self.IV)

        # Total
        self.model_data["degrees_of_freedom_total"] = self.n - 1

        ### Mean Square
        # Model (MSR)
        self.model_data["msr"] = self.model_data["sum_of_square_model"] * (1/self.model_data["degrees_of_freedom_model"])

        # Residual (error; MSE)
        self.model_data["mse"] = self.model_data["sum_of_square_residual"] * (1/self.model_data["degrees_of_freedom_residual"])

        #Total (MST)
        self.model_data["mst"] = self.model_data["sum_of_square_total"] * (1/self.model_data["degrees_of_freedom_total"])

        ## Root Mean Square Error
        self.model_data["root_mse"] = float(np.sqrt(self.model_data["mse"]))


        ### F-values
        # Model
        self.model_data["f_value_model"] = float(self.model_data["msr"] / self.model_data["mse"])
        self.model_data["f_p_value_model"] = scipy.stats.f.sf( self.model_data["f_value_model"], self.model_data["degrees_of_freedom_model"], self.model_data["degrees_of_freedom_residual"])


        ### Effect Size Measures
        # Model
        self.model_data["r squared"] = (self.model_data["sum_of_square_model"] / self.model_data["sum_of_square_total"])
        self.model_data["r squared adj."] = 1 - (self.model_data["degrees_of_freedom_total"] / self.model_data["degrees_of_freedom_residual"]) * (
            self.model_data["sum_of_square_residual"] / self.model_data["sum_of_square_total"])
        self.model_data["Eta squared"] = self.model_data["r squared"]

        self.model_data["Epsilon squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"])

        self.model_data["Omega squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"] + self.model_data["mse"])


    def _regression_base_table(self):

        self._CoreModel__regression_base_table()


    def __table_model_anova(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Constructing the model's metadata table (description), and anova table
        descriptives = {
            "Number of obs": self.n,
            "Root MSE": round(self.model_data["root_mse"], self._table_decimals.get("Root MSE", 4)),
            "R-squared": round(self.model_data["r squared"], self._table_decimals.get("R-squared", 4)),
            "Adj R-squared": round(self.model_data["r squared adj."], self._table_decimals.get("Adj R-squared", 4))
        }

        top = {
            "Source": ["Model"],
            "Sum of Squares": [round(self.model_data["sum_of_square_model"], self._table_decimals.get("Sum of Squares", 4))],
            "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"],
                                         self._table_decimals.get("Degrees of Freedom", 4))],
            "Mean Squares": [round(self.model_data["msr"], self._table_decimals.get("Mean Squares", 4))],
            "F value": [round(self.model_data["f_value_model"], self._table_decimals.get("test_stat", 4))],
            "p-value": [round(self.model_data["f_p_value_model"], self._table_decimals.get("test_stat_p", 4))],
            "Eta squared": [round(self.model_data["Eta squared"], self._table_decimals.get("Effect size", 4))],
            "Epsilon squared": [
                round(self.model_data["Epsilon squared"], self._table_decimals.get("Effect size", 4))],
            "Omega squared": [round(self.model_data["Omega squared"], self._table_decimals.get("Effect size", 4))]
        }

        bottom = {
            "Source": ["Residual", "Total"],
            "Sum of Squares": [
                round(self.model_data["sum_of_square_residual"], self._table_decimals.get("Sum of Squares", 4)),
                round(self.model_data["sum_of_square_total"], self._table_decimals.get("Sum of Squares", 4))],
            "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"],
                                         self._table_decimals.get("Degrees of Freedom", 4)),
                                   round(self.model_data["degrees_of_freedom_total"],
                                         self._table_decimals.get("Degrees of Freedom", 4))],
            "Mean Squares": [round(self.model_data["mse"], self._table_decimals.get("Mean Squares", 4)),
                             round(self.model_data["mst"], self._table_decimals.get("Mean Squares", 4))],
            "F value": ['', ''],
            "p-value": ['', ''],
            "Eta squared": ['', ''],
            "Epsilon squared": ['', ''],
            "Omega squared": ['', '']
        }

        if not pretty_format:
            bottom["F value"]= [np.nan, np.nan]
            bottom["p-value"]= [np.nan, np.nan]
            bottom["Eta squared"]= [np.nan, np.nan]
            bottom["Epsilon squared"]= [np.nan, np.nan]
            bottom["Omega squared"]= [np.nan, np.nan]


        model_results = {
            "Source": top["Source"] + bottom["Source"],
            "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
            "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
            "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
            "F value": top["F value"] + bottom["F value"],
            "p-value": top["p-value"] + bottom["p-value"],
            "Eta squared": top["Eta squared"] + bottom["Eta squared"],
            "Epsilon squared": top["Epsilon squared"] + bottom["Epsilon squared"],
            "Omega squared": top["Omega squared"] + bottom["Omega squared"]
        }

        if return_type == "Dataframe":
            descriptives_df = pd.DataFrame.from_dict(descriptives, orient="index")
            model_results_df = pd.DataFrame.from_dict(model_results)
            return descriptives_df.T, model_results_df
        else:
            return descriptives, model_results


    def __table_regression_results(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Build the coefficient table using parent's private method (returns Pandas DataFrame)
        self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, table_decimals=self._table_decimals)




    def _get_results(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):
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


        # Build the model description, and anova table
        descriptives, model_anova_table = self.__table_model_anova(return_type=return_type,
                                                                   pretty_format=pretty_format,
                                                                   table_decimals=self._table_decimals)

        # Build the coefficient table
        self.__table_regression_results(return_type=return_type,
                                        pretty_format=pretty_format,
                                        table_decimals=self._table_decimals)


        if return_type.lower() in ["dataframe", "df", "pandas.dataframe"]:
            return descriptives, model_anova_table, self.regression_table_info

        elif return_type.lower() in ["dictionary", "dict"]:
            return descriptives.to_dict(), model_anova_table.to_dict(), self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")



    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the OLS summary header with an ANOVA source table.

        Shows model name followed by a Source/SS/df/MS table with Model, Residual,
        and Total rows.

        Parameters
        ----------
        width : int
            Total character width of the output.
        model_summary_df : DataFrame or None
            Model summary DataFrame from ``self.results()``.  When None the
            ANOVA mini-table is skipped (used by Anova subclass).

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        if model_summary_df is None:
            return [self._get_model_display_name()]

        lines = []
        model_display = self._get_model_display_name()

        # Define column widths for ANOVA table
        col_w = {'source': 10, 'ss': 12, 'df': 4, 'ms': 12}
        table_w = col_w['source'] + col_w['ss'] + col_w['df'] + col_w['ms'] + 6

        def fmt_num(val, w, d=4):
            try:
                return f"{float(val):>{w}.{d}f}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        def fmt_int(val, w):
            try:
                return f"{int(val):>{w}}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        sep = "-" * col_w['source'] + "+" + "-" * (table_w - col_w['source'] - 1)

        lines.append(f"{model_display:<{table_w}}")
        lines.append(
            f"{'Source':>{col_w['source']}} | "
            f"{'SS':>{col_w['ss']}} "
            f"{'df':>{col_w['df']}} "
            f"{'MS':>{col_w['ms']}}"
        )
        lines.append(sep)
        lines.append(
            f"{'Model':>{col_w['source']}} | "
            f"{fmt_num(self.model_data.get('sum_of_square_model', 0), col_w['ss'])} "
            f"{fmt_int(self.model_data.get('degrees_of_freedom_model', 0), col_w['df'])} "
            f"{fmt_num(self.model_data.get('msr', 0), col_w['ms'])}"
        )
        lines.append(
            f"{'Residual':>{col_w['source']}} | "
            f"{fmt_num(self.model_data.get('sum_of_square_residual', 0), col_w['ss'])} "
            f"{fmt_int(self.model_data.get('degrees_of_freedom_residual', 0), col_w['df'])} "
            f"{fmt_num(self.model_data.get('mse', 0), col_w['ms'])}"
        )
        lines.append(sep)
        lines.append(
            f"{'Total':>{col_w['source']}} | "
            f"{fmt_num(self.model_data.get('sum_of_square_total', 0), col_w['ss'])} "
            f"{fmt_int(self.model_data.get('degrees_of_freedom_total', 0), col_w['df'])} "
            f"{fmt_num(self.model_data.get('mst', 0), col_w['ms'])}"
        )

        return lines


    def _splice_f_stat_lines(self):
        """
        Return the F-statistic and Prob > F lines for the header right side.

        These are not stored in the descriptives DataFrame, so they are
        built from ``self.model_data`` and spliced into the header.
        Shared by OLS and Anova header_right methods.

        Returns
        -------
        list of str
        """
        df_model = self.model_data.get("degrees_of_freedom_model", 0)
        df_resid = self.model_data.get("degrees_of_freedom_residual", 0)
        return [
            f"F({df_model}, {df_resid}) = "
            f"{self.model_data.get('f_value_model', 0):>8.2f}",
            f"Prob > F      = "
            f"{self.model_data.get('f_p_value_model', 0):>8.4f}",
        ]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the OLS summary header with fit statistics.

        When *descriptives_df* is provided the values are read from that
        DataFrame via ``to_string(header=False)`` so the summary is driven
        entirely by the DataFrames returned from ``self.results()``.

        Parameters
        ----------
        width : int
            Available character width.
        descriptives_df : DataFrame or None
            Descriptives DataFrame from ``self.results()`` (transposed,
            single-row with stat names as columns).

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        if descriptives_df is not None:
            # OLS descriptives_df is transposed: single row, columns = stat names.
            # Convert to index-oriented for to_string.
            desc_lines = descriptives_df.T.to_string(header=False).split("\n")
            return [desc_lines[0]] + self._splice_f_stat_lines() + desc_lines[1:]

        # Fallback: build from self.model_data directly
        return [
            f"Number of obs = {self.n:>8}",
        ] + self._splice_f_stat_lines() + [
            f"R-squared     = {self.model_data.get('r squared', 0):>8.4f}",
            f"Adj R-squared = {self.model_data.get('r squared adj.', 0):>8.4f}",
            f"Root MSE      = {self.model_data.get('root_mse', 0):>8.4f}",
        ]


    def _get_summary_parts(self):
        """
        Return (model_description_df, coef_df) for the OLS summary.

        Calls ``self.results()`` with stdout suppressed (to avoid side-effect
        prints) and unpacks the 3-tuple into the two DataFrames that
        ``CoreModel.summary()`` needs.

        Returns
        -------
        tuple of (model_description_df, coef_df)
        """
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model_description_df, model_summary_df, coef_df = self.results(
                return_type="Dataframe", pretty_format=True
            )
        return model_summary_df, model_description_df, coef_df
