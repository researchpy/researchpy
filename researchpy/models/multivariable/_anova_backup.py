# -*- coding: utf-8 -*-
"""
Analysis of Variance (ANOVA)

This module provides the Anova class for conducting analysis of variance
using sum of squares types I, II, and III.
"""

import re

import numpy as np
import scipy.stats
import patsy
import pandas as pd

from researchpy.models.multivariable.ols import OLS
from researchpy.utility import rounder, patsy_term_cleaner


class Anova(OLS):
    """
    Analysis of Variance (ANOVA) for factorial designs.

    This class extends OLS to provide ANOVA functionality including:
    - Type I (Sequential) sum of squares
    - Type II (Hierarchical/Partially sequential) sum of squares
    - Type III (Marginal/Orthogonal) sum of squares
    - Partial effect size measures (Eta squared, Epsilon squared, Omega squared)

    Parameters
    ----------
    formula_like : str
        A string representing a valid Patsy formula (e.g., "y ~ C(factor1) + C(factor2)").
        See https://patsy.readthedocs.io/en/latest/ for formula syntax.
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    sum_of_squares : int, optional
        Type of sum of squares to calculate. Options are:
        - 1 or "I": Sequential sum of squares
        - 2 or "II": Hierarchical/Partially sequential sum of squares
        - 3 or "III": Marginal/Orthogonal sum of squares (default)

    Attributes
    ----------
    factor_effects : dict
        Dictionary containing factor-level results including:
        - 'Source': Factor names
        - 'Sum of Squares': Sum of squares for each factor
        - 'Degrees of Freedom': Degrees of freedom for each factor
        - 'Mean Squares': Mean squares for each factor
        - 'F value': F-statistics for each factor
        - 'p-value': p-values for F-tests
        - 'Eta squared': Partial eta squared effect sizes
        - 'Epsilon squared': Partial epsilon squared effect sizes
        - 'Omega squared': Partial omega squared effect sizes

    Notes
    -----
    **Type I (Sequential) Sum of Squares:**
    Calculated by measuring the reduction in the model's overall sum of square
    error by subtracting the sum of square error for each term from the sum of
    square error from the model before.

    **Type II (Hierarchical) Sum of Squares:**
    The reduction in residual error due to adding the term to the model after
    all other terms except those that contain it.

    **Type III (Marginal) Sum of Squares:**
    Gives the sum of squares that would be obtained for each factor if it were
    to be entered into the model last. Best for models with interactions.

    Examples
    --------
    >>> import researchpy as rp
    >>> import pd as pd
    >>> df = pd.DataFrame({
    ...     'y': [1, 2, 3, 4, 5, 6],
    ...     'group': ['A', 'A', 'B', 'B', 'C', 'C']
    ... })
    >>> model = rp.Anova("y ~ C(group)", data=df)
    >>> model.results()

    See Also
    --------
    OLS : Ordinary Least Squares regression (parent class)
    """


    def __init__(self, formula_like, data=None, sum_of_squares=3):

        if data is None:
            data = {}

        super().__init__(formula_like, data)
        self.__name__ = "Researchpy.Anova"

        ###########
        factor_effects = {"Source": [],
                          "Sum of Squares": [],
                          "Degrees of Freedom": [],
                          "Mean Squares": [],
                          "F value": [],
                          "p-value": [],
                          "Eta squared": [],
                          "Epsilon squared" : [],
                          "Omega squared": []}

        ###########################################################
        #               CODUCTING THE TEST                        #
        ###########################################################
        if sum_of_squares in ["I", 1]:

            """
            Type I sum of squares is also known as sequential sum of squares and is calculated by
                measuring the reduction in the model's overall sum of square error by subtracting
                the sum of square error for each term from the sum of square error from the model
                before. The difference between the two is the sum of square factor effect for the
                current term.
            """

            previous_sum_of_squares_error = self.model_data["sum_of_square_total"]
            terms_to_include = []

            for term in self._IV_design_info.term_names:
                if term.strip().upper() == "INTERCEPT":
                    terms_to_include.append(term)
                    continue

                else:
                    if term not in terms_to_include:
                        terms_to_include.append(term)

                    # Getting design matrix set up
                    design_info2 = self._IV_design_info.subset(
                        terms_to_include)
                    x = np.asarray(patsy.build_design_matrices(
                        [design_info2], data))[0]

                    # Hat matrix
                    h = x @ np.linalg.inv(x.T @ x) @ x.T

                    # Predicted y values
                    predicted_y = h @ self.DV

                    # Calculation of residuals (error)
                    residuals = self.DV - predicted_y

                    ### Error sum of squares (SSE)
                    sum_of_square_residual = residuals.T @ residuals

                    # Calculating the Factor Effect Sum of Squares(SSR, and SSE)
                    sum_of_square_factor = previous_sum_of_squares_error - sum_of_square_residual

                    ### Degrees of freedom - Factor
                    term_subset = self._IV_design_info.subset(term)
                    term_design = np.asarray(
                        patsy.build_design_matrices([term_subset], data))[0]

                    degrees_of_freedom_factor = np.linalg.matrix_rank(
                        term_design) - 1

                    ### Mean Square - Factor (MSR-F)
                    msr_f = sum_of_square_factor * \
                        (1 / degrees_of_freedom_factor)

                    ### F-values
                    # Model
                    f_value_model = msr_f / self.model_data["mse"]
                    f_p_value_model = scipy.stats.f.sf(
                        f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    
                    ### Partial Effect Size Measures
                    eta_squared_partial = sum_of_square_factor / \
                        (sum_of_square_factor
                         + self.model_data["sum_of_square_residual"])
                        
                    epsilon_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                        (sum_of_square_factor  + self.model_data["sum_of_square_total"])
                    
                    omega_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                        ((degrees_of_freedom_factor * msr_f) + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])

                    
                    # Updating items
                    factor_effects["Source"].append(term)
                    factor_effects["Sum of Squares"].append(
                        float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(
                        float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared_partial))
                    
                    factor_effects["Epsilon squared"].append(float(epsilon_squared_partial))
                    
                    factor_effects["Omega squared"].append(float(omega_squared_partial))
                    #factor_effects["r squared"].append("")
                    #factor_effects["r squared adj."].append("")

                    # Setting new Sum of Square Residual
                    previous_sum_of_squares_error = sum_of_square_residual

                self.factor_effects = factor_effects

        elif sum_of_squares in ["II", 2]:
            """
            Type II sum of square is also known as hierarchical or
                partially sequential. This type of sum of square is
                the reduction in residual error due to adding the term
                to the model after all other terms except those
                that contain it.
            """

            for current_term in self._IV_design_info.term_names:
                terms_in_model = self._IV_design_info.term_names

                if current_term.strip().upper() == "INTERCEPT":
                    continue

                else:
                    for term in self._IV_design_info.term_names:
                        if term.strip().upper() == "INTERCEPT":
                            continue

                        elif current_term in term:
                            terms_in_model.remove(term)

                        else:
                            continue

                    if current_term.strip().upper() == "INTERCEPT":
                        continue

                    else:
                        compare_model = [item for item in terms_in_model]
                        compare_model.append(current_term)

                # Getting design matrix set up
                design_info2 = self._IV_design_info.subset(terms_in_model)
                x = np.asarray(patsy.build_design_matrices(
                    [design_info2], data[~data.isna().any(axis= 1)]))[0]

                design_info3 = self._IV_design_info.subset(compare_model)
                x_c = np.asarray(
                    patsy.build_design_matrices([design_info3], data[~data.isna().any(axis= 1)]))[0]




                # Hat matrix
                h = x @ np.linalg.inv(x.T @ x) @ x.T
                h_c = x_c @ np.linalg.inv(x_c.T @ x_c) @ x_c.T

                # Predicted y values
                predicted_y = h @ self.DV
                predicted_y_c = h_c @ self.DV

                # Calculation of residuals (error)
                residuals = self.DV - predicted_y
                residuals_c = self.DV - predicted_y_c

                ###  Error sum of squares (SSE)
                sum_of_square_residual = residuals.T @ residuals
                sum_of_square_residual_c = residuals_c.T @ residuals_c

                # Calculating the Factor Effect
                sum_of_square_factor = sum_of_square_residual - sum_of_square_residual_c

                ### Degrees of freedom - Factor
                current_term_subset = self._IV_design_info.subset(current_term)
                current_term_design = np.asarray(
                    patsy.build_design_matrices([current_term_subset], data))[0]

                degrees_of_freedom_factor = np.linalg.matrix_rank(
                    current_term_design) - 1

                ### Mean Square - Factor (MSR-F)
                msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)

                ### F-values
                # Model
                f_value_model = msr_f / self.model_data["mse"]
                f_p_value_model = scipy.stats.f.sf(
                    f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    
                ### Partial Effect Size Measures
                eta_squared_partial = sum_of_square_factor / \
                    (sum_of_square_factor
                     + self.model_data["sum_of_square_residual"])
                    
                epsilon_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    (sum_of_square_factor  + self.model_data["sum_of_square_total"])
                    
                omega_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    ((degrees_of_freedom_factor * msr_f) + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])

                
                # Updating items
                try:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared_partial))
                    factor_effects["Epsilon squared"].append(float(epsilon_squared_partial))
                    factor_effects["Omega squared"].append(float(omega_squared_partial))

                except:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float((sum_of_square_factor).item()))
                    factor_effects["Degrees of Freedom"].append(float((degrees_of_freedom_factor).item()))
                    factor_effects["Mean Squares"].append(float((msr_f).item()))
                    factor_effects["F value"].append(float((f_value_model).item()))
                    factor_effects["p-value"].append(float((f_p_value_model).item()))

                    factor_effects["Eta squared"].append(float((eta_squared_partial).item()))
                    factor_effects["Epsilon squared"].append(float((epsilon_squared_partial).item()))
                    factor_effects["Omega squared"].append(float((omega_squared_partial).item()))

            self.factor_effects = factor_effects

        elif sum_of_squares in ["III", 3]:

            """
            Type III sum of square is also known as marginal or orthogonal. This
                calculation of sum of squares give the sum of squares that would be
                obtained for each factor if it were to be entered into the model last.

            Best for if using an interaction (each main effect is able to be interpereted
                although not worthwhile if the interaction is significant)
            """

            ## This handles specification of reference category
            reference_pattern = re.compile(r'(?<=,|\s)(Treatment\(.*\))(?=\))')

            the_terms_3 = []
            for term in self._IV_design_info.term_names:

                if "Treatment" in term:
                    split_terms = term.split(":")

                    if len(split_terms) == 1:
                        the_terms_3.append(
                            re.sub(reference_pattern, 'Sum', split_terms[0]))

                    else:
                        interaction_terms = []

                        for intterm in split_terms:
                            if "Treatment" in intterm:
                                interaction_terms.append(
                                    re.sub(reference_pattern, 'Sum', intterm))
                            else:
                                interaction_terms.append(
                                    intterm.replace(")", ", Sum)"))

                        the_terms_3.append(':'.join(interaction_terms))

                else:
                    the_terms_3.append(term.replace(")", ", Sum)"))


            # Re-fitting the model using the correct specifications
            full_model = self._DV_design_info.term_names[0] + \
                " ~ " + " + ".join(the_terms_3[1:])
            y, x_full = patsy.dmatrices(full_model, data, eval_env=1)
            
                    
            terms_design_info = []
            for term in x_full.design_info.term_names[1:]:
                terms_design_info.append(x_full.design_info.subset(term))


            for current_term in the_terms_3:

                terms_in_model = []
                for term in self._IV_design_info.term_names:
                    if "Treatment" in term:
                        split_terms = term.split(":")

                        if len(split_terms) == 1:
                            terms_in_model.append(
                                re.sub(reference_pattern, 'Sum', split_terms[0]))

                        else:
                            interaction_terms = []

                            for intterm in split_terms:
                                if "Treatment" in intterm:
                                    interaction_terms.append(
                                        re.sub(reference_pattern, 'Sum', intterm))

                                else:
                                    interaction_terms.append(
                                        intterm.replace(")", ", Sum)"))

                            terms_in_model.append(':'.join(interaction_terms))

                    else:
                        terms_in_model.append(term.replace(")", ", Sum)"))

                compare_model = self._IV_design_info.term_names

                if current_term.strip().upper() == "INTERCEPT":
                    continue
                else:
                    terms_in_model.remove(current_term)                

                
                # Building Design Matrices
                design_info2 = x_full.design_info.subset(terms_in_model)
                
                x = np.asarray(patsy.build_design_matrices(
                    [design_info2], data))[0]
                

                # Hat matrix
                h = x @ np.linalg.inv(x.T @ x) @ x.T

                # Predicted y values
                predicted_y = h @ self.DV

                # Calculation of residuals (error)
                residuals = self.DV - predicted_y

                ###  Error sum of squares (SSE)
                sum_of_square_residual = residuals.T @ residuals

                # Calculating the Factor Effect
                sum_of_square_factor = sum_of_square_residual - \
                    self.model_data["sum_of_square_residual"]

                ### Degrees of freedom - Factor
                current_term_subset = x_full.design_info.subset(current_term)
                current_term_design = np.asarray(
                    patsy.build_design_matrices([current_term_subset], data))[0]

                degrees_of_freedom_factor = np.linalg.matrix_rank(
                    current_term_design) - 1
                

                ### Mean Square - Factor (MSR-F)
                msr_f = sum_of_square_factor * (1 / degrees_of_freedom_factor)
                


                ### F-values
                # Model
                f_value_model = msr_f / self.model_data["mse"]
                f_p_value_model = scipy.stats.f.sf(
                    f_value_model, degrees_of_freedom_factor, self.model_data["degrees_of_freedom_residual"])

                    
                ### Partial Effect Size Measures
                eta_squared_partial = sum_of_square_factor / \
                    (sum_of_square_factor
                     + self.model_data["sum_of_square_residual"])
                    
                epsilon_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    (sum_of_square_factor  + self.model_data["sum_of_square_total"])
                    
                omega_squared_partial = (degrees_of_freedom_factor * (msr_f - self.model_data["mse"])) / \
                    ((degrees_of_freedom_factor * msr_f) + (self.nobs - degrees_of_freedom_factor) * self.model_data["mse"])
                

                
                # Updating items
                try:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float(sum_of_square_factor))
                    factor_effects["Degrees of Freedom"].append(float(degrees_of_freedom_factor))
                    factor_effects["Mean Squares"].append(float(msr_f))
                    factor_effects["F value"].append(float(f_value_model))
                    factor_effects["p-value"].append(float(f_p_value_model))

                    factor_effects["Eta squared"].append(float(eta_squared_partial))
                    factor_effects["Epsilon squared"].append(float(epsilon_squared_partial))
                    factor_effects["Omega squared"].append(float(omega_squared_partial))

                except:
                    factor_effects["Source"].append(current_term)
                    factor_effects["Sum of Squares"].append(float((sum_of_square_factor).item()))
                    factor_effects["Degrees of Freedom"].append(float((degrees_of_freedom_factor).item()))
                    factor_effects["Mean Squares"].append(float((msr_f).item()))
                    factor_effects["F value"].append(float((f_value_model).item()))
                    factor_effects["p-value"].append(float((f_p_value_model).item()))

                    factor_effects["Eta squared"].append(float((eta_squared_partial).item()))
                    factor_effects["Epsilon squared"].append(float((epsilon_squared_partial).item()))
                    factor_effects["Omega squared"].append(float((omega_squared_partial).item()))

                self.factor_effects = factor_effects
                

    def results(self, return_type="Dataframe", decimals=4, pretty_format=True):

        if pretty_format == True:

            descriptives = {

                    "Number of obs = ": self.nobs,
                    "Root MSE = ": round(self.model_data["root_mse"], decimals),
                    "R-squared = ": round(self.model_data["r squared"], decimals),
                    "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

                }

            top = {

                    "Source": ["Model", ''],
                    "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals), ''],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals), ''],
                    "Mean Squares": [round(self.model_data["msr"], decimals), ''],
                    "F value": [round(self.model_data["f_value_model"], decimals), ''],
                    "p-value": [round(self.model_data["f_p_value_model"], decimals), ''],
                    "Eta squared": [round(self.model_data["Eta squared"], decimals), ''],
                    "Epsilon squared": [round(self.model_data["Epsilon squared"], decimals), ''],
                    "Omega squared": [round(self.model_data["Omega squared"], decimals), '']

                    }

            factors = self.factor_effects.copy()
            factors["Source"] = [patsy_term_cleaner(
                term) for term in self._IV_design_info.term_names[1:]]
            rounder(factors["Sum of Squares"], decimals=decimals)
            rounder(factors["Degrees of Freedom"], decimals=decimals)
            rounder(factors["Mean Squares"], decimals=decimals)
            rounder(factors["F value"], decimals=decimals)
            rounder(factors["p-value"], decimals=decimals)
            rounder(factors["Eta squared"], decimals=decimals)
            rounder(factors["Epsilon squared"], decimals=decimals)
            rounder(factors["Omega squared"], decimals=decimals)

            bottom = {

                    "Source": ['', "Residual", "Total"],
                    "Sum of Squares": ['', round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom": ['', round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares": ['', round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value": ['', '', ''],
                    "p-value": ['', '', ''],
                    "Eta squared": ['', '', ''],
                    "Epsilon squared" : ['', '', ''],
                    "Omega squared": ['', '', '']

                    }

            results = {

                    "Source": top["Source"] + factors["Source"] + bottom["Source"],
                    "Sum of Squares": top["Sum of Squares"] + factors["Sum of Squares"] + bottom["Sum of Squares"],
                    "Degrees of Freedom": top["Degrees of Freedom"] + factors["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares": top["Mean Squares"] + factors["Mean Squares"] + bottom["Mean Squares"],
                    "F value": top["F value"] + factors["F value"] + bottom["F value"],
                    "p-value": top["p-value"] + factors["p-value"] + bottom["p-value"],
                    "Eta squared": top["Eta squared"] + factors["Eta squared"] + bottom["Eta squared"],
                    "Epsilon squared": top["Epsilon squared"] + factors["Epsilon squared"] + bottom["Epsilon squared"],
                    "Omega squared": top["Omega squared"] + factors["Omega squared"] + bottom["Omega squared"]

                    }

        else:

            descriptives = {

                    "Number of obs = ": self.nobs,
                    "Root MSE = ": round(self.model_data["root_mse"], decimals),
                    "R-squared = ": round(self.model_data["r squared"], decimals),
                    "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

                }

            top = {

                    "Source": ["Model"],
                    "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals)],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals)],
                    "Mean Squares": [round(self.model_data["msr"], decimals)],
                    "F value": [round(self.model_data["f_value_model"], decimals)],
                    "p-value": [round(self.model_data["f_p_value_model"], decimals)]

                    }

            factors = self.factor_effects.copy()
            patsy_term_cleaner(factors["Source"])
            rounder(factors["Sum of Squares"], decimals=decimals)
            rounder(factors["Degrees of Freedom"], decimals=decimals)
            rounder(factors["Mean Squares"], decimals=decimals)
            rounder(factors["F value"], decimals=decimals)
            rounder(factors["p-value"], decimals=decimals)

            bottom = {

                    "Source": ["Residual", "Total"],
                    "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares": [round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value": [np.nan, np.nan],
                    "p-value": [np.nan, np.nan]

                    }

            results = {

                    "Source": top["Source"] + factors["Source"] + bottom["Source"],
                    "Sum of Squares": top["Sum of Squares"] + factors["Sum of Squares"] + bottom["Sum of Squares"],
                    "Degrees of Freedom": top["Degrees of Freedom"] + factors["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares": top["Mean Squares"] + factors["Mean Squares"] + bottom["Mean Squares"],
                    "F value": top["F value"] + factors["F value"] + bottom["F value"],
                    "p-value": top["p-value"] + factors["p-value"] + bottom["p-value"]

                    }

        if return_type == "Dataframe":

            print("\n"*2, "Note: Effect size values for factors are partial.", "\n"*2)
            return (pd.DataFrame.from_dict(descriptives, orient="index"), pd.DataFrame.from_dict(results))

        elif return_type == "Dictionary":

            print("\n"*2, "Note: Effect size values for factors are partial.", "\n"*2)
            return (descriptives, results)

        else:

            print(
                "Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")

    def regression_table(self, return_type="Dataframe", pretty_format=True,
                         decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                                   "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                                   'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                         *args):

        return super().results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)[2]


    def predict(self, estimate=None):
        return super().predict(estimate=estimate)


    # ------------------------------------------------------------------ #
    #                   summary() overrides for ANOVA                     #
    # ------------------------------------------------------------------ #
    def summary(self, total_width=78, return_string=False):
        """
        Print a formatted ANOVA summary to the terminal.

        Retrieves the model results as Pandas DataFrames via ``self.results()``
        and converts them into a clean, aligned text summary with:

        - Header with model name (left) and fit statistics (right)
        - ANOVA table with Model, factor, Residual, and Total rows

        Parameters
        ----------
        total_width : int, optional
            Character width for the summary output. Default is 78.
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False (prints to terminal).

        Returns
        -------
        str or None
            If return_string is True, returns the formatted summary string.
            Otherwise, prints to terminal and returns None.
        """
        # Retrieve results as DataFrames, suppressing the "Note:" side-effect
        # print that Anova.results() emits.
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            descriptives_df, results_df = self.results(
                return_type="Dataframe", pretty_format=True, decimals=4
            )

        output_lines = []

        # === HEADER SECTION ===
        # Left: model name  |  Right: fit statistics from descriptives DataFrame
        output_lines.append(
            self._summary_header(total_width)
        )

        # === ANOVA TABLE (body) — built from the results DataFrame ===
        output_lines.append(self._summary_coef_table(results_df, total_width))

        summary_str = "\n".join(output_lines)

        if return_string:
            return summary_str
        else:
            print(summary_str)
            return None


    # ------------------------------------------------------------------ #
    #               Header helpers (left / right / compose)               #
    # ------------------------------------------------------------------ #
    def _summary_header(self, width=78, model_summary_df=None:
        """
        Compose the two-column header from left and right sub-sections.

        Parameters
        ----------
        width : int
            Total character width.
        model_summary_df : DataFrame or None
            Descriptives DataFrame from ``self.results()``.  When provided the
            right-side statistics are read from this DataFrame rather than from
            ``self.model_data`` directly.
        """
        left_lines = self._summary_header_left(width)
        right_lines = self._summary_header_right(
            width, descriptives_df=model_summary_df
        )

        max_lines = max(len(left_lines), len(right_lines))
        while len(left_lines) < max_lines:
            left_lines.append("")
        while len(right_lines) < max_lines:
            right_lines.append("")

        left_width = width * 55 // 100
        gap = "    "

        combined = []
        for left_text, right_text in zip(left_lines, right_lines):
            combined.append(f"{left_text:<{left_width}}{gap}{right_text}")

        return "\n".join(combined)


    def _summary_header_left(self, width=78):
        """
        Build the left side of the Anova summary header.

        Returns just the model display name — the ANOVA source table
        is shown in the body instead.

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        return [self._get_model_display_name()]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the Anova summary header with fit statistics.

        When *descriptives_df* is provided (a single-row DataFrame whose
        columns are the statistic names), values are read from that DataFrame
        so the summary is driven entirely by the DataFrames returned from
        ``self.results()``.

        Parameters
        ----------
        width : int
            Available character width.
        descriptives_df : DataFrame or None
            Descriptives DataFrame from ``self.results()``.

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        if descriptives_df is not None:
            # The descriptives DF is index-oriented: stat names as index,
            # values in column 0.
            def _val(idx_label):
                try:
                    v = descriptives_df.loc[idx_label, 0]
                    if isinstance(v, (int, float)):
                        return f"{v:>8.4f}"
                    return f"{v:>8}"
                except (KeyError, IndexError):
                    return "     N/A"

            df_model = self.model_data.get("degrees_of_freedom_model", 0)
            df_resid = self.model_data.get("degrees_of_freedom_residual", 0)

            return [
                f"Number of obs = {_val('Number of obs = ')}",
                f"F({df_model}, {df_resid}) = {self.model_data.get('f_value_model', 0):>8.2f}",
                f"Prob > F      = {self.model_data.get('f_p_value_model', 0):>8.4f}",
                f"R-squared     = {_val('R-squared = ')}",
                f"Adj R-squared = {_val('Adj R-squared = ')}",
                f"Root MSE      = {_val('Root MSE = ')}",
            ]

        # Fallback: build from self.model_data (inherited OLS behaviour)
        return super()._summary_header_right(width)


    # ------------------------------------------------------------------ #
    #                    ANOVA table (body section)                        #
    # ------------------------------------------------------------------ #
    def _summary_coef_table(self, df_results, width=78):
        """
        Build the ANOVA table as the summary body (overrides the coefficient
        table that OLS/CoreModel would normally show).

        When *df_results* is a Pandas DataFrame (the ANOVA table returned by
        ``self.results()``), the table is built directly from its rows.
        Otherwise falls back to ``self.model_data`` / ``self.factor_effects``.

        Displays Model, individual factor rows, Residual, and Total with
        SS, df, MS, F, p-value, Eta², Epsilon², Omega².

        Parameters
        ----------
        df_results : DataFrame or None
            The ANOVA results DataFrame from ``self.results()``.
        width : int
            Total character width of the output.

        Returns
        -------
        str
            Formatted ANOVA table string.
        """
        rows = self._build_anova_rows(df_results)

        # Column definitions: (key, header_label, col_width, decimals)
        columns = [
            ("ss",     "SS",        12, 4),
            ("df",     "df",         4, 0),
            ("ms",     "MS",        12, 4),
            ("f",      "F",          9, 4),
            ("p",      "p-value",    8, 4),
            ("eta",    "Eta^2",      8, 4),
            ("eps",    "Eps^2",      8, 4),
            ("omega",  "Omega^2",    8, 4),
        ]

        source_w = 14
        sep_char = "-"

        # --- header line ---
        header = f"{'Source':>{source_w}} |"
        for _key, label, col_w, _ in columns:
            header += f" {label:>{col_w}}"
        lines = [sep_char * width, header]

        # --- mid separator ---
        lines.append(
            sep_char * source_w + "+" + sep_char * (width - source_w - 1)
        )

        # --- helper formatter ---
        def fmt(val, w, d):
            if val is None or val == "":
                return " " * w
            try:
                if d == 0:
                    return f"{int(val):>{w}}"
                return f"{float(val):>{w}.{d}f}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        # --- data rows ---
        for row in rows:
            line = f"{row['source']:>{source_w}} |"
            for key, _, col_w, d in columns:
                line += f" {fmt(row.get(key), col_w, d)}"
            lines.append(line)

        lines.append(sep_char * width)

        # Note about partial effect sizes
        lines.append("Note: Effect size values for factors are partial.")

        return "\n".join(lines)


    # ------------------------------------------------------------------ #
    #                  Row assembly for the ANOVA table                    #
    # ------------------------------------------------------------------ #
    def _build_anova_rows(self, df_results=None):
        """
        Assemble a list of row dicts for the ANOVA summary table.

        When *df_results* is a Pandas DataFrame (from ``self.results()``),
        the rows are read directly from the DataFrame.  Otherwise falls back
        to ``self.model_data`` / ``self.factor_effects``.

        Row order: Model, [each factor], Residual, Total.

        Parameters
        ----------
        df_results : DataFrame or None
            The ANOVA results DataFrame from ``self.results()``.

        Returns
        -------
        list of dict
            Each dict has keys: source, ss, df, ms, f, p, eta, eps, omega.
        """
        # --- DataFrame path: read rows directly from the results DF ---
        if df_results is not None and isinstance(df_results, pd.DataFrame):
            return self._build_anova_rows_from_df(df_results)

        # --- Fallback path: build from self.model_data / self.factor_effects ---
        return self._build_anova_rows_from_model_data()


    def _build_anova_rows_from_df(self, df):
        """
        Build ANOVA row dicts by iterating over the results DataFrame.

        The DataFrame columns are expected to match the output of
        ``self.results(return_type="Dataframe", pretty_format=True)``:

            Source | Sum of Squares | Degrees of Freedom | Mean Squares |
            F value | p-value | Eta squared | Epsilon squared | Omega squared

        Empty-string or NaN cells are normalised to ``None`` so that the
        formatter renders blanks for Residual / Total effect-size columns.

        Parameters
        ----------
        df : DataFrame
            ANOVA results table.

        Returns
        -------
        list of dict
        """
        rows = []

        def _clean(val):
            """Return None for empty / NaN values, else the value."""
            if val is None:
                return None
            if isinstance(val, str) and val.strip() == "":
                return None
            try:
                if pd.isna(val):
                    return None
            except (TypeError, ValueError):
                pass
            return val

        for _, row in df.iterrows():
            source = _clean(row.get("Source"))
            if source is None:
                continue  # skip blank separator rows

            rows.append({
                "source": str(source),
                "ss":     _clean(row.get("Sum of Squares")),
                "df":     _clean(row.get("Degrees of Freedom")),
                "ms":     _clean(row.get("Mean Squares")),
                "f":      _clean(row.get("F value")),
                "p":      _clean(row.get("p-value")),
                "eta":    _clean(row.get("Eta squared")),
                "eps":    _clean(row.get("Epsilon squared")),
                "omega":  _clean(row.get("Omega squared")),
            })

        return rows


    def _build_anova_rows_from_model_data(self):
        """
        Fallback: build ANOVA row dicts from ``self.model_data`` and
        ``self.factor_effects`` when no DataFrame is available.

        Returns
        -------
        list of dict
        """
        rows = []

        # --- Model row ---
        rows.append({
            "source": "Model",
            "ss":    self.model_data.get("sum_of_square_model"),
            "df":    self.model_data.get("degrees_of_freedom_model"),
            "ms":    self.model_data.get("msr"),
            "f":     self.model_data.get("f_value_model"),
            "p":     self.model_data.get("f_p_value_model"),
            "eta":   self.model_data.get("Eta squared"),
            "eps":   self.model_data.get("Epsilon squared"),
            "omega": self.model_data.get("Omega squared"),
        })

        # --- Factor rows ---
        if hasattr(self, "factor_effects"):
            n_factors = len(self.factor_effects["Sum of Squares"])
            cleaned_names = [
                patsy_term_cleaner(t)
                for t in self._IV_design_info.term_names[1:]
            ]
            for i in range(n_factors):
                name = (cleaned_names[i]
                        if i < len(cleaned_names)
                        else self.factor_effects["Source"][i])
                rows.append({
                    "source": name,
                    "ss":    self.factor_effects["Sum of Squares"][i],
                    "df":    self.factor_effects["Degrees of Freedom"][i],
                    "ms":    self.factor_effects["Mean Squares"][i],
                    "f":     self.factor_effects["F value"][i],
                    "p":     self.factor_effects["p-value"][i],
                    "eta":   self.factor_effects["Eta squared"][i],
                    "eps":   self.factor_effects["Epsilon squared"][i],
                    "omega": self.factor_effects["Omega squared"][i],
                })

        # --- Residual row ---
        rows.append({
            "source": "Residual",
            "ss":    self.model_data.get("sum_of_square_residual"),
            "df":    self.model_data.get("degrees_of_freedom_residual"),
            "ms":    self.model_data.get("mse"),
            "f": None, "p": None, "eta": None, "eps": None, "omega": None,
        })

        # --- Total row ---
        rows.append({
            "source": "Total",
            "ss":    self.model_data.get("sum_of_square_total"),
            "df":    self.model_data.get("degrees_of_freedom_total"),
            "ms":    self.model_data.get("mst"),
            "f": None, "p": None, "eta": None, "eps": None, "omega": None,
        })

        return rows


