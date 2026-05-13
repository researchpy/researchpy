import patsy
import pandas as pd

from researchpy.models.base import CoreModel
from researchpy.core.containerclasses import ModelResults
from researchpy.utility import *


class LinearModel(CoreModel):
    """

    This is a subclass of core_model for linear statistical models that use ordinary least squares.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95, display_summary=True,
                 family="gaussian", link="normal", solver_method="ols", obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True},
                 table_decimals=None):

        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_method=solver_method, obj_function=obj_function,
                         table_decimals=table_decimals)

        self.__name__ = "Researchpy.LinearModel"

        # OLS fit to compute the coefficients (betas) — stored in self.CoefResults.betas
        self._CoreModel__ols_fit()

        # Compute the model sum of squares, degrees of freedom, mean squares, F-value, p-value,
        # and effect size measures — stored in self.ModelEffects
        self.__model_sum_of_square_stats()

        # Compute standard errors and confidence intervals — stored in self.CoefResults
        self.__compute_beta_se_and_stats()



    def _compute_model_stats(self):
        predicted_y = self.IV @ self.CoefResults.betas  # predicted y values
        residuals = self.DV - predicted_y  # Calculation of residuals (error)

        J = self._j_matrix(to_return=True)   # Creating the J matrix
        I = self._identity_matrix(to_return=True)  # Creating the identity matrix

        ### Sum of Squares
        # Total sum of squares (SSTO)
        self.ModelEffects.sum_of_square_total = float(
            (self.DV.T @ self.DV - (1 / self.n) * self.DV.T @ J @ self.DV).item()
        )

        # Model sum of squares (SSR)
        self.ModelEffects.sum_of_square_model = float(
            (self.CoefResults.betas.T @ self.IV.T @ self.DV - (1 / self.n) * self.DV.T @ J @ self.DV).item()
        )

        # Error sum of squares (SSE)
        self.ModelEffects.sum_of_square_residual = float((residuals.T @ residuals).item())

        ### Degrees of freedom
        # Model
        self.ModelEffects.degrees_of_freedom_model = np.linalg.matrix_rank(self.IV) - 1

        # Error
        self.ModelEffects.degrees_of_freedom_residual = self.n - np.linalg.matrix_rank(self.IV)

        # Total
        self.ModelEffects.degrees_of_freedom_total = self.n - 1

        ### Mean Square
        # Model (MSR)
        self.ModelEffects.msr = self.ModelEffects.sum_of_square_model * (1 / self.ModelEffects.degrees_of_freedom_model)

        # Residual (error; MSE)
        self.ModelEffects.mse = self.ModelEffects.sum_of_square_residual * (1 / self.ModelEffects.degrees_of_freedom_residual)

        # Total (MST)
        self.ModelEffects.mst = self.ModelEffects.sum_of_square_total * (1 / self.ModelEffects.degrees_of_freedom_total)

        ## Root Mean Square Error
        self.ModelEffects.root_mse = float(np.sqrt(self.ModelEffects.mse))

        ### F-values
        # Model
        self.ModelEffects.model_test_stat = float(self.ModelEffects.msr / self.ModelEffects.mse)
        self.ModelEffects.model_test_pval = scipy.stats.f.sf(self.ModelEffects.model_test_stat,
                                                              self.ModelEffects.degrees_of_freedom_model,
                                                              self.ModelEffects.degrees_of_freedom_residual)

        ### Effect Size Measures
        # Model
        self.ModelEffects.r_squared = (self.ModelEffects.sum_of_square_model / self.ModelEffects.sum_of_square_total)
        self.ModelEffects.r_squared_adj = 1 - (
                    self.ModelEffects.degrees_of_freedom_total / self.ModelEffects.degrees_of_freedom_residual) * (
                                                    self.ModelEffects.sum_of_square_residual / self.ModelEffects.sum_of_square_total)
        self.ModelEffects.eta_squared = self.ModelEffects.r_squared

        self.ModelEffects.epsilon_squared = (self.ModelEffects.degrees_of_freedom_model * (
                    self.ModelEffects.msr - self.ModelEffects.mse)) / (self.ModelEffects.sum_of_square_total)

        self.ModelEffects.omega_squared = (self.ModelEffects.degrees_of_freedom_model * (
                    self.ModelEffects.msr - self.ModelEffects.mse)) / (
                                                       self.ModelEffects.sum_of_square_total + self.ModelEffects.mse)



    # Compute standard errors and statistics
    def __compute_beta_se_and_stats(self):
        variance_covariance_beta_matrix = self.__variance_covariance_beta_matrix(method="standard", to_return=True, add_to_self=False)

        ## Standard Errors
        self.CoefResults.std_error = np.sqrt(np.diag(variance_covariance_beta_matrix)).reshape(-1, 1)

        ## T-statistics
        self.CoefResults.test_stat = self.CoefResults.betas * (1 / self.CoefResults.std_error)

        ## Two-sided p-value
        self.CoefResults.p_value = np.array([
            float((scipy.stats.t.sf(np.abs(t), self.ModelEffects.degrees_of_freedom_residual) * 2).item())
            for t in self.CoefResults.test_stat
        ])

        self._CoreModel__compute_confidence_intervals()


    def __variance_covariance_residual_matrix(self, method="standard", to_return=True, add_to_self=False):

        if method == "standard":
            H = self._hat_matrix(to_return=True)
            I = self._identity_matrix(to_return=True)
            variance_covariance_residual_matrix = np.asarray(self.ModelEffects.mse * (I - H))

        if add_to_self:
            self.variance_covariance_residual_matrix = variance_covariance_residual_matrix

        if to_return:
            return variance_covariance_residual_matrix


    def __variance_covariance_beta_matrix(self, method="standard", to_return=True, add_to_self=False):

        if method == "standard":
            try:
                variance_covariance_beta_matrix = np.asarray(self.ModelEffects.mse * np.linalg.inv(self.IV.T @ self.IV))
            except np.linalg.LinAlgError:
                variance_covariance_beta_matrix = np.asarray(self.ModelEffects.mse * np.linalg.pinv(self.IV.T @ self.IV))

        if add_to_self:
            self.variance_covariance_beta_matrix = variance_covariance_beta_matrix

        if to_return:
            return variance_covariance_beta_matrix




    def __model_sum_of_square_stats(self):

        predicted_y = self.IV @ self.CoefResults.betas     # predicted y values
        residuals = self.DV - predicted_y                   # Calculation of residuals (error)

        J = self._j_matrix(to_return=True)                  # Creating the J matrix
        I = self._identity_matrix(to_return=True)           # Creating the identity matrix


        ### Sum of Squares
        # Total sum of squares (SSTO)
        self.ModelEffects.sum_of_square_total = float((self.DV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Model sum of squares (SSR)
        self.ModelEffects.sum_of_square_model = float((self.CoefResults.betas.T @ self.IV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Error sum of squares (SSE)
        self.ModelEffects.sum_of_square_residual = float((residuals.T @ residuals).item())

        ### Degrees of freedom
        # Model
        self.ModelEffects.degrees_of_freedom_model = return_numeric(np.linalg.matrix_rank(self.IV) - 1)

        # Error
        self.ModelEffects.degrees_of_freedom_residual = return_numeric(self.n - np.linalg.matrix_rank(self.IV))

        # Total
        self.ModelEffects.degrees_of_freedom_total = return_numeric(self.n - 1)

        ### Mean Square
        # Model (MSR)
        self.ModelEffects.msr = return_numeric(self.ModelEffects.sum_of_square_model * (1/self.ModelEffects.degrees_of_freedom_model))

        # Residual (error; MSE)
        self.ModelEffects.mse = return_numeric(self.ModelEffects.sum_of_square_residual * (1/self.ModelEffects.degrees_of_freedom_residual))

        #Total (MST)
        self.ModelEffects.mst = return_numeric(self.ModelEffects.sum_of_square_total * (1/self.ModelEffects.degrees_of_freedom_total))

        ## Root Mean Square Error
        self.ModelEffects.root_mse = float(np.sqrt(self.ModelEffects.mse))


        ### F-values
        # Model
        self.ModelEffects.model_test_stat = float(self.ModelEffects.msr / self.ModelEffects.mse)
        self.ModelEffects.model_test_pval = return_numeric(scipy.stats.f.sf(self.ModelEffects.model_test_stat,
                                                                            self.ModelEffects.degrees_of_freedom_model,
                                                                            self.ModelEffects.degrees_of_freedom_residual)
                                                           )


        ### Effect Size Measures
        # Model
        self.ModelEffects.r_squared = return_numeric(self.ModelEffects.sum_of_square_model / self.ModelEffects.sum_of_square_total)
        self.ModelEffects.r_squared_adj = 1 - (self.ModelEffects.degrees_of_freedom_total / self.ModelEffects.degrees_of_freedom_residual) * (
            self.ModelEffects.sum_of_square_residual / self.ModelEffects.sum_of_square_total)
        self.ModelEffects.eta_squared = self.ModelEffects.r_squared

        self.ModelEffects.epsilon_squared = (self.ModelEffects.degrees_of_freedom_model * (self.ModelEffects.msr - self.ModelEffects.mse)) / (self.ModelEffects.sum_of_square_total)

        self.ModelEffects.omega_squared = (self.ModelEffects.degrees_of_freedom_model * (self.ModelEffects.msr - self.ModelEffects.mse)) / (self.ModelEffects.sum_of_square_total + self.ModelEffects.mse)


    def _regression_base_table(self):

        self._CoreModel__regression_base_table()


    def __table_model_anova(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        me = self.ModelEffects

        # Constructing the model's metadata table (description), and anova table
        descriptives = {
            "Number of obs": self.n,
            "Root MSE": round(me.root_mse, self._table_decimals.get("Root MSE", 4)),
            "R-squared": round(me.r_squared, self._table_decimals.get("R-squared", 4)),
            "Adj R-squared": round(me.r_squared_adj, self._table_decimals.get("Adj R-squared", 4))
        }

        top = {
            "Source": ["Model"],
            "Sum of Squares": [round(me.sum_of_square_model, self._table_decimals.get("Sum of Squares", 4))],
            "Degrees of Freedom": [round(me.degrees_of_freedom_model,
                                         self._table_decimals.get("Degrees of Freedom", 4))],
            "Mean Squares": [round(me.msr, self._table_decimals.get("Mean Squares", 4))],
            "F value": [round(me.model_test_stat, self._table_decimals.get("test_stat", 4))],
            "p-value": [round(me.model_test_pval, self._table_decimals.get("test_stat_p", 4))],
            "Eta squared": [round(me.eta_squared, self._table_decimals.get("Effect size", 4))],
            "Epsilon squared": [
                round(me.epsilon_squared, self._table_decimals.get("Effect size", 4))],
            "Omega squared": [round(me.omega_squared, self._table_decimals.get("Effect size", 4))]
        }

        bottom = {
            "Source": ["Residual", "Total"],
            "Sum of Squares": [
                round(me.sum_of_square_residual, self._table_decimals.get("Sum of Squares", 4)),
                round(me.sum_of_square_total, self._table_decimals.get("Sum of Squares", 4))],
            "Degrees of Freedom": [round(me.degrees_of_freedom_residual,
                                         self._table_decimals.get("Degrees of Freedom", 4)),
                                   round(me.degrees_of_freedom_total,
                                         self._table_decimals.get("Degrees of Freedom", 4))],
            "Mean Squares": [round(me.mse, self._table_decimals.get("Mean Squares", 4)),
                             round(me.mst, self._table_decimals.get("Mean Squares", 4))],
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


    def __table_fit_statistics(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        me = self.ModelEffects

        # Constructing the model's metadata table (description), and anova table
        df_model = round(me.degrees_of_freedom_model,
                         self._table_decimals.get("Degrees of Freedom", 1))
        df_resid = round(me.degrees_of_freedom_residual,
                         self._table_decimals.get("Degrees of Freedom", 1))
        test_stat_model = round(me.model_test_stat, self._table_decimals.get("test_stat", 4))
        test_pval_model = round(me.model_test_pval, self._table_decimals.get("test_stat_p", 4))
        fit_statistics = {
            "nobs": [f"N = {self.n}"],
            "model_test_stat": [f"F({df_model}, {df_resid}) = {test_stat_model}"],
            "model_test_pval": [f"Prob > F = {test_pval_model}"],
            "rootmse": [f"Root MSE = {round(me.root_mse, self._table_decimals.get('Root MSE', 4))}"],
            "r_squared": [
                f"R-squared = {round(me.r_squared, self._table_decimals.get('R-squared', 4))}"],
            "r_squared_adj": [f"Adj R-squared = {round(me.r_squared_adj, self._table_decimals.get('Adj R-squared', 4))}"]
        }


        '''fit_statistics = {
            "nobs": ["N observations = ", f"{self.n}"],
            "model_test_stat": [f"F({df_model}, {df_resid}) = ", f"{test_stat_model}"],
            "model_test_pval": [f"Prob > F = ", f"{test_pval_model}"],
            "rootmse": [f"Root MSE = ", f"{round(me.root_mse, self._table_decimals.get('Root MSE', 4))}"],
            "r_squared": [f"R-squared = ", f"{round(me.r_squared, self._table_decimals.get('R-squared', 4))}"],
            "r_squared_adj": [f"Adj R-squared = ", f"{round(me.r_squared_adj, self._table_decimals.get('Adj R-squared', 4))}"]
        }'''


        if return_type == "Dataframe":
            fit_statistics_df = pd.DataFrame.from_dict(fit_statistics, orient="index").rename(columns={0: 'Model Fit Statistics'})

            #fit_statistics_df = pd.DataFrame.from_dict(fit_statistics, orient="index").rename(columns={0: 'Model', 1: 'Statistics'})
            return fit_statistics_df
        else:
            return fit_statistics


    def __table_model(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        me = self.ModelEffects
        blank = '' if pretty_format else np.nan

        # Constructing the model's metadata table (description), and anova table
        top = {
            "Source": ["Model"],
            "Sum of Squares": [round(me.sum_of_square_model, self._table_decimals.get("Sum of Squares", 4))],
            "Degrees of Freedom": [round(me.degrees_of_freedom_model,
                                         self._table_decimals.get("Degrees of Freedom", 4))],
            "Mean Squares": [round(me.msr, self._table_decimals.get("Mean Squares", 4))],
            "Eta squared": [round(me.eta_squared, self._table_decimals.get("Effect size", 4))],
            "Epsilon squared": [
                round(me.epsilon_squared, self._table_decimals.get("Effect size", 4))],
            "Omega squared": [round(me.omega_squared, self._table_decimals.get("Effect size", 4))]
        }

        bottom = {
            "Source": ["Residual", "Total"],
            "Sum of Squares": [
                round(me.sum_of_square_residual, self._table_decimals.get("Sum of Squares", 4)),
                round(me.sum_of_square_total, self._table_decimals.get("Sum of Squares", 4))],
            "Degrees of Freedom": [round(me.degrees_of_freedom_residual,
                                         self._table_decimals.get("Degrees of Freedom", 4)),
                                   round(me.degrees_of_freedom_total,
                                         self._table_decimals.get("Degrees of Freedom", 4))],
            "Mean Squares": [round(me.mse, self._table_decimals.get("Mean Squares", 4)),
                             round(me.mst, self._table_decimals.get("Mean Squares", 4))],
            "Eta squared": [blank, blank],
            "Epsilon squared": [blank, blank],
            "Omega squared": [blank, blank]
        }


        model_results = {
            "Source": top["Source"] + bottom["Source"],
            "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
            "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
            "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
            "Eta squared": top["Eta squared"] + bottom["Eta squared"],
            "Epsilon squared": top["Epsilon squared"] + bottom["Epsilon squared"],
            "Omega squared": top["Omega squared"] + bottom["Omega squared"]
        }

        if return_type == "Dataframe":
            model_results_df = pd.DataFrame.from_dict(model_results)
            return model_results_df
        else:
            return model_results


    def __table_regression_results(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Build the coefficient table using parent's private method (returns Pandas DataFrame)
        return self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, table_decimals=self._table_decimals)


    def _get_ModelResults(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):
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
        if return_type.lower() not in ["dataframe", "df", "pandas.dataframe", "dictionary", "dict"]:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals


        # Build the model description, and anova table
        fit_statistics, model_table = self.__table_model_anova(return_type=return_type,
                                                               pretty_format=pretty_format,
                                                               table_decimals=self._table_decimals)

        fit_statistics = self.__table_fit_statistics(return_type=return_type,
                                                     pretty_format=pretty_format,
                                                     table_decimals=self._table_decimals)

        model_table = self.__table_model(return_type=return_type,
                                         pretty_format=pretty_format,
                                         table_decimals=self._table_decimals)

        # Build the coefficient table
        coefficients = self.__table_regression_results(return_type=return_type,
                                                       pretty_format=pretty_format,
                                                       table_decimals=self._table_decimals)

        self.ModelResults = ModelResults(
            model_name=self._get_model_display_name(),
            fit_statistics=fit_statistics,
            model_table=model_table,
            coefficients=coefficients,
        )

        return self.ModelResults


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

        mr = self._get_ModelResults(return_type=return_type,
                                    pretty_format=pretty_format,
                                    table_decimals=self._table_decimals)

        return mr.fit_statistics, mr.model_table, mr.coefficients





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

        me = self.ModelEffects
        lines.append(
            f"{'Model':>{col_w['source']}} | "
            f"{fmt_num(me.sum_of_square_model or 0, col_w['ss'])} "
            f"{fmt_int(me.degrees_of_freedom_model or 0, col_w['df'])} "
            f"{fmt_num(me.msr or 0, col_w['ms'])}"
        )
        lines.append(
            f"{'Residual':>{col_w['source']}} | "
            f"{fmt_num(me.sum_of_square_residual or 0, col_w['ss'])} "
            f"{fmt_int(me.degrees_of_freedom_residual or 0, col_w['df'])} "
            f"{fmt_num(me.mse or 0, col_w['ms'])}"
        )
        lines.append(sep)
        lines.append(
            f"{'Total':>{col_w['source']}} | "
            f"{fmt_num(me.sum_of_square_total or 0, col_w['ss'])} "
            f"{fmt_int(me.degrees_of_freedom_total or 0, col_w['df'])} "
            f"{fmt_num(me.mst or 0, col_w['ms'])}"
        )

        return lines


    def _splice_f_stat_lines(self):
        """
        Return the F-statistic and Prob > F lines for the header right side.

        These are not stored in the descriptives DataFrame, so they are
        built from ``self.ModelEffects`` and spliced into the header.
        Shared by OLS and Anova header_right methods.

        Returns
        -------
        list of str
        """
        me = self.ModelEffects
        df_model = me.degrees_of_freedom_model or 0
        df_resid = me.degrees_of_freedom_residual or 0
        return [
            f"F({df_model}, {df_resid}) = "
            f"{(me.model_test_stat or 0):>8.2f}",
            f"Prob > F      = "
            f"{(me.model_test_pval or 0):>8.4f}",
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
            desc_lines = descriptives_df.to_string(header=False, index=False).split("\n")
            #return [desc_lines[0]] + self._splice_f_stat_lines() + desc_lines[1:]
            return desc_lines

        # Fallback: build from self.ModelEffects directly
        me = self.ModelEffects
        return ([f"Number of obs = {self.n:>8}",] + self._splice_f_stat_lines() +
                [f"R-squared     = {(me.r_squared or 0):>8.4f}",
                 f"Adj R-squared = {(me.r_squared_adj or 0):>8.4f}",
                 f"Root MSE      = {(me.root_mse or 0):>8.4f}",
                 ])

