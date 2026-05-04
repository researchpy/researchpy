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


    def summary(self, return_string=False, show_coefficients=False):
        """
        Print a formatted summary of the ANOVA results to the terminal.

        This method displays the ANOVA results in a Stata-inspired format with:
        - Model header information (model type, number of observations, fit statistics)
        - ANOVA table with factor effects, sum of squares, F-values, and effect sizes

        Parameters
        ----------
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False (prints to terminal).
        show_coefficients : bool, optional
            If True, also displays the coefficient table (like OLS).
            Default is False (shows only ANOVA table).

        Returns
        -------
        str or None
            If return_string is True, returns the formatted summary string.
            Otherwise, prints to terminal and returns None.

        Examples
        --------
        >>> model = rp.Anova("y ~ C(group)", data=df)
        >>> model.summary()  # prints ANOVA table to terminal
        >>> model.summary(show_coefficients=True)  # also shows coefficient table
        """
        # Get the results - use pretty_format=True to get properly formatted data
        # Temporarily suppress the note that results() prints
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            results = self.results(return_type="Dictionary", pretty_format=True)
        finally:
            sys.stdout = old_stdout
        descriptives, anova_table = results

        # Build the summary string
        output_lines = []
        total_width = 90  # Wider for ANOVA table with effect sizes

        # === HEADER SECTION ===
        output_lines.append(self._summary_header(total_width))
        output_lines.append("")

        # === ANOVA TABLE ===
        output_lines.append(self._summary_anova_table(anova_table, total_width))

        # === OPTIONAL COEFFICIENT TABLE ===
        if show_coefficients:
            output_lines.append("")
            output_lines.append("Coefficients:")
            # Get coefficient table from parent OLS
            coef_results = super().results(return_type="Dictionary")
            output_lines.append(self._summary_coef_table(coef_results, total_width))

        # Note about effect sizes
        output_lines.append("")
        output_lines.append("Note: Effect size values for factors are partial.")

        # Join all sections
        summary_str = "\n".join(output_lines)

        if return_string:
            return summary_str
        else:
            print(summary_str)
            return None


    def _summary_anova_table(self, anova_data, width=90):
        """
        Build the ANOVA table section of the summary output.

        Parameters
        ----------
        anova_data : dict
            The ANOVA results dictionary with Source, SS, df, MS, F, p-value, effect sizes
        width : int
            Total width of the output

        Returns
        -------
        str
            Formatted ANOVA table
        """
        lines = []

        # Check if we have effect size columns
        has_effect_sizes = "Eta squared" in anova_data

        # Define column widths
        if has_effect_sizes:
            col_widths = {
                'source': 14,
                'ss': 12,
                'df': 4,
                'ms': 12,
                'f': 9,
                'p': 8,
                'eta': 8,
                'eps': 8,
                'omega': 8
            }
        else:
            col_widths = {
                'source': 14,
                'ss': 14,
                'df': 6,
                'ms': 14,
                'f': 12,
                'p': 10
            }

        # Header separator
        sep_line = "-" * width
        mid_sep = "-" * col_widths['source'] + "+" + "-" * (width - col_widths['source'] - 1)

        lines.append(sep_line)

        # Column header line
        if has_effect_sizes:
            header = (f"{'Source':>{col_widths['source']}} | "
                      f"{'SS':>{col_widths['ss']}} "
                      f"{'df':>{col_widths['df']}} "
                      f"{'MS':>{col_widths['ms']}} "
                      f"{'F':>{col_widths['f']}} "
                      f"{'p-value':>{col_widths['p']}} "
                      f"{'Eta^2':>{col_widths['eta']}} "
                      f"{'Eps^2':>{col_widths['eps']}} "
                      f"{'Omega^2':>{col_widths['omega']}}")
        else:
            header = (f"{'Source':>{col_widths['source']}} | "
                      f"{'Sum of Squares':>{col_widths['ss']}} "
                      f"{'df':>{col_widths['df']}} "
                      f"{'Mean Squares':>{col_widths['ms']}} "
                      f"{'F':>{col_widths['f']}} "
                      f"{'p-value':>{col_widths['p']}}")

        lines.append(header)
        lines.append(mid_sep)

        # Format numeric values helper
        def fmt_num(val, width, decimals=4):
            if val == "" or val is None:
                return " " * width
            if isinstance(val, float) and np.isnan(val):
                return " " * width
            try:
                return f"{float(val):>{width}.{decimals}f}"
            except (ValueError, TypeError):
                return f"{str(val):>{width}}"

        # Get data rows
        sources = anova_data.get("Source", [])
        ss_vals = anova_data.get("Sum of Squares", [])
        df_vals = anova_data.get("Degrees of Freedom", [])
        ms_vals = anova_data.get("Mean Squares", [])
        f_vals = anova_data.get("F value", [])
        p_vals = anova_data.get("p-value", [])
        
        if has_effect_sizes:
            eta_vals = anova_data.get("Eta squared", [])
            eps_vals = anova_data.get("Epsilon squared", [])
            omega_vals = anova_data.get("Omega squared", [])

        for i, source in enumerate(sources):
            # Skip empty source rows (used for spacing in pretty_format)
            if source == "" or source is None:
                continue
                
            ss = ss_vals[i] if i < len(ss_vals) else ""
            df = df_vals[i] if i < len(df_vals) else ""
            ms = ms_vals[i] if i < len(ms_vals) else ""
            f = f_vals[i] if i < len(f_vals) else ""
            p = p_vals[i] if i < len(p_vals) else ""

            # Format df as integer if it's a number
            if df != "" and df is not None:
                try:
                    df_str = f"{int(float(df)):>{col_widths['df']}}"
                except (ValueError, TypeError):
                    df_str = f"{str(df):>{col_widths['df']}}"
            else:
                df_str = " " * col_widths['df']

            if has_effect_sizes:
                eta = eta_vals[i] if i < len(eta_vals) else ""
                eps = eps_vals[i] if i < len(eps_vals) else ""
                omega = omega_vals[i] if i < len(omega_vals) else ""

                row = (f"{str(source):>{col_widths['source']}} | "
                       f"{fmt_num(ss, col_widths['ss'], 4)} "
                       f"{df_str} "
                       f"{fmt_num(ms, col_widths['ms'], 4)} "
                       f"{fmt_num(f, col_widths['f'], 4)} "
                       f"{fmt_num(p, col_widths['p'], 4)} "
                       f"{fmt_num(eta, col_widths['eta'], 4)} "
                       f"{fmt_num(eps, col_widths['eps'], 4)} "
                       f"{fmt_num(omega, col_widths['omega'], 4)}")
            else:
                row = (f"{str(source):>{col_widths['source']}} | "
                       f"{fmt_num(ss, col_widths['ss'], 4)} "
                       f"{df_str} "
                       f"{fmt_num(ms, col_widths['ms'], 4)} "
                       f"{fmt_num(f, col_widths['f'], 4)} "
                       f"{fmt_num(p, col_widths['p'], 4)}")

            lines.append(row)

        lines.append(sep_line)

        return "\n".join(lines)

